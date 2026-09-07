"""Column-name hygiene: strip control characters, make names unique, keep PyArrow happy."""

from __future__ import annotations

import re

import pandas as pd

from atml.config import load_config
from atml.dtypes import text_columns
from atml.validation.report import Severity, ValidationReport

_CONTROL = re.compile(r"[\x00-\x1f\x7f<>]")


def sanitize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """Return a copy with safe, unique, string column names plus a report of what changed.

    Turkish letters are kept on purpose; only control chars, angle brackets (HTML
    injection into Streamlit markdown), leading/trailing whitespace and duplicates are fixed.
    """
    cfg = load_config()["validation"]
    report = ValidationReport()
    seen: dict[str, int] = {}
    new_cols: list[str] = []

    for idx, col in enumerate(df.columns):
        original = str(col)
        name = _CONTROL.sub("", original).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"column_{idx}"
            report.add(Severity.WARNING, f"Unnamed column renamed to '{name}'", original)
        if len(name) > cfg["max_column_name_length"]:
            name = name[: cfg["max_column_name_length"]]
            report.add(Severity.INFO, "Column name truncated", original)
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
            report.add(Severity.WARNING, f"Duplicate column renamed to '{name}'", original)
        else:
            seen[name] = 0
        if name != original:
            report.add(Severity.INFO, f"'{original}' -> '{name}'", original)
        new_cols.append(name)

    out = df.copy()
    out.columns = new_cols
    # Mixed-type object columns break Arrow serialisation in Streamlit; cast to str.
    for col in text_columns(out):
        if out[col].map(type).nunique() > 1:
            out[col] = out[col].astype(str).replace({"nan": None})
            report.add(Severity.INFO, "Mixed-type column cast to string", col)
    return out, report
