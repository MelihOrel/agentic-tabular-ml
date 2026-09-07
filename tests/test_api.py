from __future__ import annotations

import importlib

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from atml.modeling import AutoML


@pytest.fixture
def client(clf_df: pd.DataFrame, tmp_path, monkeypatch):
    path = AutoML().fit(clf_df, "target").save(tmp_path / "champion.joblib")
    monkeypatch.setenv("ATML_MODEL_PATH", str(path))
    import api.main as main

    importlib.reload(main)
    return TestClient(main.app), clf_df


def test_health(client) -> None:
    c, _ = client
    assert c.get("/health").json()["status"] == "ok"


def test_model_info(client) -> None:
    c, _ = client
    body = c.get("/model").json()
    assert body["task"] == "classification"
    assert body["target"] == "target"


def test_predict(client) -> None:
    c, df = client
    rows = df.drop(columns=["target"]).head(3).to_dict(orient="records")
    body = c.post("/predict", json={"rows": rows}).json()
    assert len(body["predictions"]) == 3
    assert len(body["confidence"]) == 3


def test_predict_missing_column_returns_422(client) -> None:
    c, df = client
    body = c.post("/predict", json={"rows": [{"x1": 0.5}]})
    assert body.status_code == 422
