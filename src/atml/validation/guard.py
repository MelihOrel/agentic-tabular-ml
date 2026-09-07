"""MLGuard: refuse to train models that cannot be meaningful.

Checks run *before* any model touches the data. Errors block training,
warnings are shown to the user (and returned to the agent as observations).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atml.config import load_config
from atml.dtypes import is_textlike
from atml.validation.report import Severity, ValidationReport


class MLGuard:
    def __init__(self, cfg: dict | None = None):
        self.cfg = (cfg or load_config())["validation"]

    def check(self, df: pd.DataFrame, target: str) -> ValidationReport:
        r = ValidationReport()
        c = self.cfg

        if target not in df.columns:
            r.add(Severity.ERROR, "Target column not found", target)
            return r
        if len(df) < c["min_rows_for_training"]:
            r.add(
                Severity.ERROR, f"Need at least {c['min_rows_for_training']} rows, have {len(df)}"
            )
        if pd.api.types.is_datetime64_any_dtype(df[target]):
            r.add(Severity.ERROR, "A datetime column cannot be a prediction target", target)

        y = df[target].dropna()
        missing_frac = 1 - len(y) / max(len(df), 1)
        if missing_frac > 0.5:
            r.add(Severity.ERROR, f"{missing_frac:.0%} of target values are missing", target)
        elif missing_frac > 0:
            r.add(Severity.WARNING, f"{missing_frac:.1%} of target rows will be dropped", target)

        if y.nunique() < 2:
            r.add(Severity.ERROR, "Target has a single value; nothing to learn", target)
        elif not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 10:
            counts = y.value_counts()
            if (counts < c["min_samples_per_class"]).any():
                rare = counts[counts < c["min_samples_per_class"]].index.tolist()
                r.add(
                    Severity.ERROR,
                    f"Classes with fewer than {c['min_samples_per_class']} samples: {rare}",
                    target,
                )
            ratio = counts.max() / counts.min() if counts.min() else np.inf
            if ratio > 10:
                r.add(Severity.WARNING, f"Class imbalance ratio {ratio:.1f}:1", target)

        features = df.drop(columns=[target])
        if features.shape[1] == 0:
            r.add(Severity.ERROR, "No feature columns left")
            return r

        for col in features.columns:
            s = features[col]
            if s.nunique(dropna=True) <= 1:
                r.add(Severity.WARNING, "Zero-variance column; will be dropped", col)
            elif is_textlike(s) and s.nunique() / max(len(s), 1) > c["high_cardinality_ratio"]:
                r.add(Severity.WARNING, "Looks like an identifier (near-unique text)", col)
            if s.isna().mean() > 0.6:
                r.add(Severity.WARNING, f"{s.isna().mean():.0%} missing", col)

        # Constant columns produce 0/0 in the correlation matrix; exclude them.
        num = features.select_dtypes(include="number")
        num = num.loc[:, num.nunique(dropna=True) > 1]
        if num.shape[1] >= 2:
            corr = num.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
            pairs = [
                (a, b, float(upper.loc[a, b]))
                for a in upper.index
                for b in upper.columns
                if pd.notna(upper.loc[a, b]) and upper.loc[a, b] > c["multicollinearity_threshold"]
            ]
            for a, b, v in pairs[:5]:
                r.add(Severity.WARNING, f"Highly correlated with '{b}' (r={v:.2f})", a)

        # Target leakage heuristic: a feature that perfectly predicts the target.
        if pd.api.types.is_numeric_dtype(y):
            for col in num.columns:
                if abs(num[col].corr(df.loc[num.index, target])) > 0.999:
                    r.add(Severity.WARNING, "Near-perfect correlation with target (leakage?)", col)
        return r
