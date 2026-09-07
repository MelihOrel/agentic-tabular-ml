"""Small, honest AutoML: few models, proper cross-validation, one held-out test set.

Design choices (deliberate, see README):
    * Three model families per task, not ten. Leaderboards with ten near-identical
      tree models mostly measure noise.
    * All preprocessing lives inside the sklearn Pipeline so there is no leakage
      and the fitted object is self-contained for the API.
    * Scores reported are mean +/- std over K folds, then a single test-set score
      for the champion. Both are kept so the gap is visible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from atml.config import load_config

Task = Literal["classification", "regression"]


def detect_task(y: pd.Series) -> Task:
    """Numeric with many distinct values -> regression; otherwise classification."""
    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        return "classification"
    n_unique = y.nunique()
    if n_unique <= 10 or (n_unique <= 20 and (y.dropna() % 1 == 0).all()):
        return "classification"
    return "regression"


def _model_zoo(task: Task, seed: int) -> dict[str, Any]:
    if task == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
            "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
            "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
        }
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=seed),
    }


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


@dataclass
class LeaderboardEntry:
    model: str
    cv_mean: float
    cv_std: float
    metric: str


@dataclass
class TrainResult:
    task: Task
    target: str
    features: list[str]
    metric: str
    leaderboard: list[LeaderboardEntry]
    champion: str
    test_scores: dict[str, float]
    feature_importance: dict[str, float]
    n_train: int
    n_test: int
    pipeline: Pipeline = field(repr=False)
    class_labels: list[Any] | None = None

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("pipeline")
        d["leaderboard"] = [asdict(e) for e in self.leaderboard]
        return d

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "meta": self.summary()}, path)
        return path


class AutoML:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or load_config()
        self.seed = int(cfg["seed"])
        self.cfg = cfg["modeling"]

    def fit(self, df: pd.DataFrame, target: str, features: list[str] | None = None) -> TrainResult:
        data = df.dropna(subset=[target])
        features = features or [c for c in data.columns if c != target]
        # Drop datetime columns and zero-variance columns; the guard already warned about them.
        features = [
            c
            for c in features
            if not pd.api.types.is_datetime64_any_dtype(data[c])
            and data[c].nunique(dropna=True) > 1
        ]
        X, y = data[features], data[target]
        task = detect_task(y)
        class_labels = None
        if task == "classification":
            y = y.astype(str)
            class_labels = sorted(y.unique().tolist())

        metric = self.cfg[
            "classification_metric" if task == "classification" else "regression_metric"
        ]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=self.cfg["test_size"],
            random_state=self.seed,
            stratify=y if task == "classification" else None,
        )
        splitter = (
            StratifiedKFold(self.cfg["cv_folds"], shuffle=True, random_state=self.seed)
            if task == "classification"
            else KFold(self.cfg["cv_folds"], shuffle=True, random_state=self.seed)
        )

        zoo = _model_zoo(task, self.seed)
        wanted = self.cfg["models"][task]
        board: list[LeaderboardEntry] = []
        pipelines: dict[str, Pipeline] = {}
        for name in wanted:
            pipe = Pipeline([("prep", build_preprocessor(X_tr)), ("model", zoo[name])])
            scores = cross_val_score(pipe, X_tr, y_tr, cv=splitter, scoring=metric, n_jobs=1)
            board.append(LeaderboardEntry(name, float(scores.mean()), float(scores.std()), metric))
            pipelines[name] = pipe

        board.sort(key=lambda e: e.cv_mean, reverse=True)
        champion = board[0].model
        best = pipelines[champion].fit(X_tr, y_tr)
        pred = best.predict(X_te)

        if task == "classification":
            test_scores = {
                "accuracy": float(accuracy_score(y_te, pred)),
                "f1_macro": float(f1_score(y_te, pred, average="macro")),
            }
        else:
            test_scores = {
                "r2": float(r2_score(y_te, pred)),
                "mae": float(mean_absolute_error(y_te, pred)),
                "rmse": float(root_mean_squared_error(y_te, pred)),
            }

        return TrainResult(
            task=task,
            target=target,
            features=features,
            metric=metric,
            leaderboard=board,
            champion=champion,
            test_scores=test_scores,
            feature_importance=self._importance(best, X_te, y_te),
            n_train=len(X_tr),
            n_test=len(X_te),
            pipeline=best,
            class_labels=class_labels,
        )

    def _importance(self, pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Permutation importance on the test set, on the *original* columns.

        Model-agnostic and unaffected by one-hot expansion, so it is comparable across
        the leaderboard. Limited repeats to keep the UI responsive.
        """
        from sklearn.inspection import permutation_importance

        res = permutation_importance(pipe, X, y, n_repeats=5, random_state=self.seed, n_jobs=1)
        imp = dict(zip(X.columns, res.importances_mean.astype(float), strict=True))
        return dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))


def load_model(path: str | Path) -> tuple[Pipeline, dict[str, Any]]:
    blob = joblib.load(path)
    return blob["pipeline"], blob["meta"]


def predict_frame(pipe: Pipeline, meta: dict[str, Any], rows: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in meta["features"] if c not in rows.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = rows[meta["features"]]
    out = rows.copy()
    out["prediction"] = pipe.predict(X)
    if meta["task"] == "classification" and hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        out["confidence"] = np.max(proba, axis=1)
    return out
