from __future__ import annotations

import pandas as pd
import pytest

from atml.modeling import AutoML, detect_task
from atml.modeling.automl import load_model, predict_frame


class TestTaskDetection:
    def test_binary_int_is_classification(self) -> None:
        assert detect_task(pd.Series([0, 1, 0, 1])) == "classification"

    def test_text_is_classification(self) -> None:
        assert detect_task(pd.Series(["a", "b", "a"])) == "classification"

    def test_continuous_is_regression(self) -> None:
        assert detect_task(pd.Series(range(500)).astype(float) / 3) == "regression"


class TestAutoML:
    def test_classification_learns_signal(self, clf_df: pd.DataFrame) -> None:
        res = AutoML().fit(clf_df, "target")
        assert res.task == "classification"
        assert res.test_scores["f1_macro"] > 0.7  # data is genuinely learnable
        assert len(res.leaderboard) == 3
        assert res.leaderboard[0].cv_mean >= res.leaderboard[-1].cv_mean

    def test_regression_learns_signal(self, reg_df: pd.DataFrame) -> None:
        res = AutoML().fit(reg_df, "price")
        assert res.task == "regression"
        assert res.test_scores["r2"] > 0.8

    def test_reproducible_across_runs(self, clf_df: pd.DataFrame) -> None:
        a = AutoML().fit(clf_df, "target")
        b = AutoML().fit(clf_df, "target")
        assert a.champion == b.champion
        assert a.test_scores == pytest.approx(b.test_scores)

    def test_important_feature_ranks_above_noise(self, clf_df: pd.DataFrame) -> None:
        df = clf_df.assign(noise=range(len(clf_df)))
        res = AutoML().fit(df, "target")
        assert res.feature_importance["x1"] > res.feature_importance["noise"]

    def test_save_load_predict_round_trip(self, clf_df: pd.DataFrame, tmp_path) -> None:
        res = AutoML().fit(clf_df, "target")
        path = res.save(tmp_path / "m.joblib")

        pipe, meta = load_model(path)
        out = predict_frame(pipe, meta, clf_df.head(5))
        assert len(out) == 5
        assert "confidence" in out
        assert (
            out["prediction"].tolist()
            == res.pipeline.predict(clf_df.head(5)[res.features]).tolist()
        )

    def test_predict_rejects_missing_columns(self, clf_df: pd.DataFrame, tmp_path) -> None:
        res = AutoML().fit(clf_df, "target")
        pipe, meta = load_model(res.save(tmp_path / "m.joblib"))
        with pytest.raises(ValueError, match="Missing feature columns"):
            predict_frame(pipe, meta, clf_df[["x1"]].head())
