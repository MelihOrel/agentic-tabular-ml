from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

SEED = 42


@pytest.fixture
def clf_df() -> pd.DataFrame:
    """Learnable classification set: target depends on the numeric features."""
    rng = np.random.default_rng(SEED)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(5, 2, n)
    cat = rng.choice(["a", "b", "c"], n)
    logit = 1.5 * x1 - 0.4 * (x2 - 5) + (cat == "a") * 1.2
    y = (logit + rng.normal(0, 0.5, n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "target": y})


@pytest.fixture
def reg_df() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = 300
    x1 = rng.normal(10, 3, n)
    x2 = rng.uniform(0, 100, n)
    y = 3.0 * x1 + 0.5 * x2 + rng.normal(0, 2, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "price": y})


@pytest.fixture
def dirty_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "value": [1.0, 2.0, np.nan, 4.0, 5.0, 1000.0],
            "grup": ["a", "a", "b", "b", None, "a"],
            "sabit": [1, 1, 1, 1, 1, 1],
            "kimlik": [f"id{i}" for i in range(6)],
        }
    )
