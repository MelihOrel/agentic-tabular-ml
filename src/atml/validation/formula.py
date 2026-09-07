"""AST-restricted formula evaluation for calculated columns.

Users (and the LLM agent) can write expressions such as ``price / area * 1.18``.
Instead of eval(), the expression is parsed into an AST and only a whitelist of
node types, operators and functions is allowed, so ``__import__('os')`` or
attribute access can never execute.
"""

from __future__ import annotations

import ast
import operator as op
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


class FormulaError(ValueError):
    """Raised when a formula uses anything outside the whitelist."""


_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {ast.USub: op.neg, ast.UAdd: op.pos}
_CMP_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Gt: op.gt,
    ast.GtE: op.ge,
    ast.Lt: op.lt,
    ast.LtE: op.le,
    ast.Eq: op.eq,
    ast.NotEq: op.ne,
}
_FUNCS: dict[str, Callable[..., Any]] = {
    "abs": np.abs,
    "log": np.log,
    "log1p": np.log1p,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "round": np.round,
    "min": np.minimum,
    "max": np.maximum,
    "where": np.where,
}


class _SafeEvaluator(ast.NodeVisitor):
    def __init__(self, columns: dict[str, pd.Series]):
        self.columns = columns

    def visit(self, node: ast.AST) -> Any:  # type: ignore[override]
        method = f"visit_{type(node).__name__}"
        if not hasattr(self, method):
            raise FormulaError(f"'{type(node).__name__}' is not allowed in formulas")
        return getattr(self, method)(node)

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise FormulaError("Only numeric constants are allowed")

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.columns:
            return self.columns[node.id]
        raise FormulaError(f"Unknown column '{node.id}'")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        fn = _BIN_OPS.get(type(node.op))
        if fn is None:
            raise FormulaError("Operator not allowed")
        return fn(self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        fn = _UNARY_OPS.get(type(node.op))
        if fn is None:
            raise FormulaError("Operator not allowed")
        return fn(self.visit(node.operand))

    def visit_Compare(self, node: ast.Compare) -> Any:
        if len(node.ops) != 1:
            raise FormulaError("Chained comparisons are not allowed")
        fn = _CMP_OPS.get(type(node.ops[0]))
        if fn is None:
            raise FormulaError("Comparison not allowed")
        return fn(self.visit(node.left), self.visit(node.comparators[0]))

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise FormulaError("Only whitelisted functions are allowed: " + ", ".join(_FUNCS))
        if node.keywords:
            raise FormulaError("Keyword arguments are not allowed")
        return _FUNCS[node.func.id](*[self.visit(a) for a in node.args])


def evaluate_formula(df: pd.DataFrame, formula: str) -> pd.Series:
    """Evaluate ``formula`` against numeric columns of ``df`` and return a Series."""
    if len(formula) > 500:
        raise FormulaError("Formula too long")
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise FormulaError(f"Syntax error: {exc.msg}") from exc

    numeric = {c: df[c] for c in df.select_dtypes(include="number").columns}
    result = _SafeEvaluator(numeric).visit(tree)
    if np.isscalar(result):
        result = pd.Series(np.full(len(df), result), index=df.index)
    return pd.Series(result, index=df.index)
