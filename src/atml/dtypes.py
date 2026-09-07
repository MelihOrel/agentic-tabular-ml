"""dtype helpers that behave the same on pandas 2 and pandas 3.

pandas 3 introduced a dedicated ``str`` dtype, so the classic ``series.dtype == object``
and ``select_dtypes(include="object")`` idioms silently stop matching text columns.
Every module goes through these two helpers instead.
"""

from __future__ import annotations

import pandas as pd


def is_textlike(s: pd.Series) -> bool:
    """True for object / string / categorical columns, on either pandas major version."""
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    ) and not pd.api.types.is_numeric_dtype(s)


def text_columns(df: pd.DataFrame) -> list[str]:
    """Names of all text-like columns, in frame order."""
    return [c for c in df.columns if is_textlike(df[c])]
