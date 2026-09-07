"""FastAPI serving layer for a model exported from the app.

    ATML_MODEL_PATH=models/champion.joblib uvicorn api.main:app --reload

The pipeline carries its own preprocessing, so the API only validates that the
feature columns are present and hands rows to ``predict_frame``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from atml.modeling.automl import load_model, predict_frame

MODEL_PATH = Path(os.getenv("ATML_MODEL_PATH", "models/champion.joblib"))

app = FastAPI(title="atml prediction API", version="0.1.0")
_state: dict[str, Any] = {}


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1, max_length=10_000)


class PredictResponse(BaseModel):
    task: str
    target: str
    predictions: list[Any]
    confidence: list[float] | None = None


def _load() -> None:
    if "pipe" not in _state:
        if not MODEL_PATH.exists():
            raise HTTPException(
                503, f"No model at {MODEL_PATH}. Train and export one from the app."
            )
        _state["pipe"], _state["meta"] = load_model(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": "pipe" in _state, "model_path": str(MODEL_PATH)}


@app.get("/model")
def model_info() -> dict[str, Any]:
    _load()
    meta = dict(_state["meta"])
    return {k: meta[k] for k in ("task", "target", "features", "champion", "test_scores", "metric")}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    _load()
    try:
        out = predict_frame(_state["pipe"], _state["meta"], pd.DataFrame(req.rows))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return PredictResponse(
        task=_state["meta"]["task"],
        target=_state["meta"]["target"],
        predictions=out["prediction"].tolist(),
        confidence=out["confidence"].round(4).tolist() if "confidence" in out else None,
    )
