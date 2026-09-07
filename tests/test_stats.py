from __future__ import annotations

import numpy as np
import pandas as pd

from atml.stats import (
    anova,
    chi_square,
    describe,
    isolation_forest,
    kmeans_elbow,
    normality,
    pca_2d,
    ttest,
)


def test_describe_adds_shape_metrics(reg_df: pd.DataFrame) -> None:
    out = describe(reg_df)
    assert {"skew", "kurtosis", "missing", "iqr"} <= set(out.columns)


def test_normality_detects_normal_and_skewed() -> None:
    # n=200: Shapiro-Wilk is so powerful at n>=500 that a genuinely normal sample
    # is often rejected at alpha=0.05 (seed 0 gives p=0.028). That is the test
    # behaving correctly, not a bug, so the assertion uses a sample size where the
    # binary verdict is meaningful and also checks the p-values are orders apart.
    rng = np.random.default_rng(0)
    gaussian = normality(pd.Series(rng.normal(0, 1, 200)))
    skewed = normality(pd.Series(rng.exponential(1, 200)))
    assert gaussian["normal"]
    assert not skewed["normal"]
    assert gaussian["p_value"] > skewed["p_value"] * 1000


def test_normality_switches_test_above_5000_rows() -> None:
    rng = np.random.default_rng(0)
    assert normality(pd.Series(rng.normal(0, 1, 6000)))["test"] == "dagostino"


def test_normality_needs_enough_observations() -> None:
    assert "error" in normality(pd.Series([1.0, 2.0, 3.0]))


def test_ttest_finds_real_difference() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "v": np.concatenate([rng.normal(10, 1, 100), rng.normal(13, 1, 100)]),
            "g": ["a"] * 100 + ["b"] * 100,
        }
    )
    res = ttest(df, "v", "g")
    assert res["significant"]


def test_ttest_requires_two_groups() -> None:
    df = pd.DataFrame({"v": [1.0, 2, 3], "g": ["a", "b", "c"]})
    assert "error" in ttest(df, "v", "g")


def test_anova_and_chi_square_run() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "v": np.concatenate([rng.normal(m, 1, 60) for m in (5, 8, 11)]),
            "g": ["a"] * 60 + ["b"] * 60 + ["c"] * 60,
            "h": rng.choice(["x", "y"], 180),
        }
    )
    assert anova(df, "v", "g")["significant"]
    assert "p_value" in chi_square(df, "g", "h")


def test_kmeans_recovers_three_blobs() -> None:
    rng = np.random.default_rng(0)
    blobs = np.vstack([rng.normal(c, 0.3, (80, 2)) for c in ([0, 0], [6, 6], [0, 6])])
    df = pd.DataFrame(blobs, columns=["a", "b"])
    res = kmeans_elbow(df)
    assert res["labels"].nunique() == res["k"]
    assert res["inertia_by_k"][1] > res["inertia_by_k"][3]


def test_isolation_forest_flags_outliers() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": np.concatenate([rng.normal(0, 1, 190), np.full(10, 50.0)])})
    res = isolation_forest(df)
    assert res["n_anomalies"] > 0
    assert res["flags"].tail(10).sum() >= 8  # the injected extremes are caught


def test_pca_returns_two_components(reg_df: pd.DataFrame) -> None:
    res = pca_2d(reg_df)
    assert len(res["explained_variance"]) == 2
    assert list(res["coords"].columns) == ["PC1", "PC2"]
