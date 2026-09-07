"""Pure DataFrame -> DataFrame cleaning operations.

Each function returns ``(new_df, description)``; the caller decides whether to
commit it to History. Nothing here mutates its input.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from atml.config import load_config
from atml.validation.formula import evaluate_formula

ImputeStrategy = Literal["mean", "median", "mode", "constant", "ffill", "bfill", "drop_rows"]
OutlierMethod = Literal["iqr", "zscore"]
OutlierAction = Literal["clip", "drop"]


def impute(
    df: pd.DataFrame, column: str, strategy: ImputeStrategy, fill_value: Any = None
) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    s = out[column]
    n_missing = int(s.isna().sum())
    if n_missing == 0:
        return out, f"'{column}' has no missing values"

    if strategy == "drop_rows":
        out = out.dropna(subset=[column])
        return out, f"Dropped {n_missing} rows with missing '{column}'"
    if strategy in ("mean", "median"):
        if not pd.api.types.is_numeric_dtype(s):
            raise ValueError(f"'{strategy}' needs a numeric column")
        value = s.mean() if strategy == "mean" else s.median()
    elif strategy == "mode":
        value = s.mode(dropna=True).iloc[0] if not s.dropna().empty else None
    elif strategy == "constant":
        if fill_value is None:
            raise ValueError("fill_value is required for constant imputation")
        value = fill_value
    elif strategy in ("ffill", "bfill"):
        out[column] = s.ffill() if strategy == "ffill" else s.bfill()
        return out, f"{strategy} filled {n_missing} values in '{column}'"
    else:
        raise ValueError(f"Unknown strategy {strategy}")

    out[column] = s.fillna(value)
    return out, f"Filled {n_missing} missing values in '{column}' with {strategy} ({value!r})"


def handle_outliers(
    df: pd.DataFrame, column: str, method: OutlierMethod = "iqr", action: OutlierAction = "clip"
) -> tuple[pd.DataFrame, str]:
    cfg = load_config()["cleaning"]["outlier"]
    out = df.copy()
    s = out[column]
    if not pd.api.types.is_numeric_dtype(s):
        raise ValueError("Outlier handling needs a numeric column")

    if method == "iqr":
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - cfg["iqr_factor"] * iqr, q3 + cfg["iqr_factor"] * iqr
    else:
        mu, sd = s.mean(), s.std(ddof=0)
        lo, hi = mu - cfg["zscore_threshold"] * sd, mu + cfg["zscore_threshold"] * sd

    mask = (s < lo) | (s > hi)
    n = int(mask.sum())
    if action == "clip":
        out[column] = s.clip(lower=lo, upper=hi)
        return out, f"Clipped {n} outliers in '{column}' to [{lo:.4g}, {hi:.4g}] ({method})"
    out = out.loc[~mask]
    return out, f"Dropped {n} outlier rows in '{column}' ({method})"


def cast_column(df: pd.DataFrame, column: str, target: str) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    before_na = out[column].isna().sum()
    if target == "numeric":
        out[column] = pd.to_numeric(out[column], errors="coerce")
    elif target == "datetime":
        out[column] = pd.to_datetime(out[column], errors="coerce")
    elif target == "category":
        out[column] = out[column].astype("category")
    elif target == "string":
        out[column] = out[column].astype("string")
    else:
        raise ValueError(f"Unknown target type {target}")
    lost = int(out[column].isna().sum() - before_na)
    return out, f"Cast '{column}' to {target}" + (f" ({lost} values became NaN)" if lost else "")


def add_calculated_column(df: pd.DataFrame, name: str, formula: str) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    out[name] = evaluate_formula(out, formula)
    return out, f"Added '{name}' = {formula}"


def apply_filter(
    df: pd.DataFrame, column: str, operator: str, value: Any
) -> tuple[pd.DataFrame, str]:
    s = df[column]
    ops = {
        "==": s == value,
        "!=": s != value,
        ">": s > value,
        ">=": s >= value,
        "<": s < value,
        "<=": s <= value,
        "contains": s.astype(str).str.contains(str(value), case=False, na=False),
        "is_null": s.isna(),
        "not_null": s.notna(),
    }
    if operator not in ops:
        raise ValueError(f"Unknown operator {operator}")
    mask = np.asarray(ops[operator], dtype=bool)
    out = df.loc[mask].copy()
    return out, f"Filter {column} {operator} {value!r}: kept {len(out)} of {len(df)} rows"
