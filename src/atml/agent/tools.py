"""Tool functions the ReAct agent can call.

They are written as plain Python callables that close over an AgentSession and
return short strings. ``build_tools`` wraps them with LangChain's ``StructuredTool``
only when LangChain is installed, so the core stays importable (and testable)
without any LLM dependency.

Guard rails:
    * ``train_model`` runs MLGuard first and refuses on errors.
    * ``add_calculated_column`` goes through the AST-safe formula evaluator.
    * No tool can read files or the network; loading is done by the human in the UI.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from atml.agent.session import AgentSession
from atml.modeling import AutoML
from atml.processing import (
    add_calculated_column,
    apply_filter,
    cast_column,
    handle_outliers,
    impute,
)
from atml.stats import describe, normality
from atml.validation import FormulaError, MLGuard


def make_tool_functions(session: AgentSession) -> dict[str, Callable[..., str]]:
    def inspect_data() -> str:
        """Describe the current dataset: shape, dtypes, missing values per column."""
        df = session.df
        info = {
            "rows": len(df),
            "columns": {
                c: {
                    "dtype": str(df[c].dtype),
                    "missing": int(df[c].isna().sum()),
                    "unique": int(df[c].nunique()),
                }
                for c in df.columns
            },
        }
        return json.dumps(info)

    def summary_statistics() -> str:
        """Numeric summary (mean, std, quartiles, skew, kurtosis, missing) as JSON."""
        return describe(session.df).round(4).to_json(orient="index")

    def check_normality(column: str) -> str:
        """Run a normality test on a numeric column."""
        return json.dumps(normality(session.df[column]))

    def impute_column(column: str, strategy: str, fill_value: Any = None) -> str:
        """Fill missing values. strategy: mean|median|mode|constant|ffill|bfill|drop_rows."""
        new, desc = impute(session.df, column, strategy, fill_value)  # type: ignore[arg-type]
        return session.commit(new, "impute", desc)

    def clip_outliers(column: str, method: str = "iqr", action: str = "clip") -> str:
        """Handle outliers on a numeric column. method: iqr|zscore, action: clip|drop."""
        new, desc = handle_outliers(session.df, column, method, action)  # type: ignore[arg-type]
        return session.commit(new, "outliers", desc)

    def convert_type(column: str, target: str) -> str:
        """Cast a column. target: numeric|datetime|category|string."""
        new, desc = cast_column(session.df, column, target)
        return session.commit(new, "cast", desc)

    def add_column(name: str, formula: str) -> str:
        """Add a calculated numeric column from a safe formula, e.g. 'price / area'."""
        try:
            new, desc = add_calculated_column(session.df, name, formula)
        except FormulaError as exc:
            return f"Formula rejected: {exc}"
        return session.commit(new, "formula", desc)

    def filter_rows(column: str, operator: str, value: Any = None) -> str:
        """Keep rows matching a condition. operator: ==,!=,>,>=,<,<=,contains,is_null,not_null."""
        new, desc = apply_filter(session.df, column, operator, value)
        return session.commit(new, "filter", desc)

    def train_model(target: str) -> str:
        """Run MLGuard, then cross-validated AutoML on the current data. Returns leaderboard."""
        report = MLGuard().check(session.df, target)
        if not report.ok:
            return "Training refused by MLGuard:\n" + report.summary()
        result = AutoML().fit(session.df, target)
        session.train_result = result
        payload = {
            "task": result.task,
            "champion": result.champion,
            "leaderboard": [
                {"model": e.model, e.metric: round(e.cv_mean, 4), "std": round(e.cv_std, 4)}
                for e in result.leaderboard
            ],
            "test_scores": {k: round(v, 4) for k, v in result.test_scores.items()},
            "top_features": dict(list(result.feature_importance.items())[:8]),
            "guard_warnings": [i.message for i in report.warnings],
        }
        return json.dumps(payload)

    def undo_last() -> str:
        """Revert the most recent operation."""
        op = session.history.undo()
        return f"Undid: {op.description}" if op else "Nothing to undo"

    return {
        "inspect_data": inspect_data,
        "summary_statistics": summary_statistics,
        "check_normality": check_normality,
        "impute_column": impute_column,
        "clip_outliers": clip_outliers,
        "convert_type": convert_type,
        "add_column": add_column,
        "filter_rows": filter_rows,
        "train_model": train_model,
        "undo_last": undo_last,
    }


def build_tools(session: AgentSession) -> list[Any]:
    """Wrap the functions as LangChain StructuredTools. Requires the [agent] extra."""
    from langchain_core.tools import StructuredTool

    return [
        StructuredTool.from_function(fn, name=name, description=fn.__doc__)
        for name, fn in make_tool_functions(session).items()
    ]
