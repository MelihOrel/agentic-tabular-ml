"""Descriptive statistics, hypothesis tests and unsupervised helpers.

Every function returns plain dicts / DataFrames so the same output can be
rendered by Streamlit or read back by the agent as an observation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from atml.config import load_config


def _alpha() -> float:
    return float(load_config()["stats"]["alpha"])


def describe(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    out = num.describe().T
    out["skew"] = num.skew()
    out["kurtosis"] = num.kurtosis()
    out["missing"] = num.isna().sum()
    out["iqr"] = out["75%"] - out["25%"]
    return out


def normality(s: pd.Series) -> dict[str, Any]:
    x = s.dropna().to_numpy(dtype=float)
    if len(x) < 8:
        return {"test": "shapiro", "n": len(x), "error": "need at least 8 observations"}
    if len(x) <= 5000:
        stat, p = stats.shapiro(x)
        name = "shapiro"
    else:
        stat, p = stats.normaltest(x)
        name = "dagostino"
    return {
        "test": name,
        "n": int(len(x)),
        "statistic": float(stat),
        "p_value": float(p),
        "normal": bool(p > _alpha()),
    }


def ttest(df: pd.DataFrame, value: str, group: str) -> dict[str, Any]:
    levels = df[group].dropna().unique()
    if len(levels) != 2:
        return {"error": f"'{group}' must have exactly 2 levels, has {len(levels)}"}
    a = df.loc[df[group] == levels[0], value].dropna()
    b = df.loc[df[group] == levels[1], value].dropna()
    stat, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "test": "welch_t",
        "groups": [str(levels[0]), str(levels[1])],
        "means": [float(a.mean()), float(b.mean())],
        "statistic": float(stat),
        "p_value": float(p),
        "significant": bool(p < _alpha()),
    }


def anova(df: pd.DataFrame, value: str, group: str) -> dict[str, Any]:
    groups = [g[value].dropna().to_numpy() for _, g in df.groupby(group, observed=True)]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return {"error": "need at least 2 groups with 2+ observations"}
    stat, p = stats.f_oneway(*groups)
    return {
        "test": "one_way_anova",
        "k": len(groups),
        "statistic": float(stat),
        "p_value": float(p),
        "significant": bool(p < _alpha()),
    }


def chi_square(df: pd.DataFrame, a: str, b: str) -> dict[str, Any]:
    table = pd.crosstab(df[a], df[b])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"error": "both columns need at least 2 categories"}
    stat, p, dof, _ = stats.chi2_contingency(table)
    return {
        "test": "chi_square",
        "statistic": float(stat),
        "p_value": float(p),
        "dof": int(dof),
        "significant": bool(p < _alpha()),
    }


def _scaled_numeric(
    df: pd.DataFrame, columns: list[str] | None = None
) -> tuple[np.ndarray, pd.Index]:
    num = df[columns] if columns else df.select_dtypes(include="number")
    num = num.dropna()
    return StandardScaler().fit_transform(num), num.index


def kmeans_elbow(
    df: pd.DataFrame, columns: list[str] | None = None, k: int | None = None
) -> dict[str, Any]:
    cfg = load_config()
    X, idx = _scaled_numeric(df, columns)
    max_k = min(cfg["stats"]["kmeans_max_k"], max(2, len(X) // 5))
    inertias = {}
    for kk in range(1, max_k + 1):
        inertias[kk] = float(KMeans(kk, n_init=10, random_state=cfg["seed"]).fit(X).inertia_)
    if k is None:
        # Elbow by largest second difference of inertia.
        vals = np.array(list(inertias.values()))
        second = np.diff(vals, 2) if len(vals) > 2 else np.array([0])
        k = int(np.argmax(second) + 2) if len(second) else 2
    labels = KMeans(k, n_init=10, random_state=cfg["seed"]).fit_predict(X)
    return {
        "k": int(k),
        "inertia_by_k": inertias,
        "labels": pd.Series(labels, index=idx, name="cluster"),
    }


def isolation_forest(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, Any]:
    cfg = load_config()
    X, idx = _scaled_numeric(df, columns)
    model = IsolationForest(
        contamination=cfg["stats"]["isolation_forest_contamination"], random_state=cfg["seed"]
    ).fit(X)
    flag = pd.Series(model.predict(X) == -1, index=idx, name="is_anomaly")
    score = pd.Series(-model.score_samples(X), index=idx, name="anomaly_score")
    return {"n_anomalies": int(flag.sum()), "flags": flag, "scores": score}


def pca_2d(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, Any]:
    cfg = load_config()
    X, idx = _scaled_numeric(df, columns)
    pca = PCA(n_components=2, random_state=cfg["seed"]).fit(X)
    coords = pd.DataFrame(pca.transform(X), index=idx, columns=["PC1", "PC2"])
    return {"explained_variance": pca.explained_variance_ratio_.round(4).tolist(), "coords": coords}
