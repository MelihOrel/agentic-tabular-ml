from __future__ import annotations

import pandas as pd
import pytest

from atml.processing import History, apply_filter, cast_column, handle_outliers, impute


class TestImpute:
    def test_median_fills_and_reports(self, dirty_df: pd.DataFrame) -> None:
        out, desc = impute(dirty_df, "value", "median")
        assert out["value"].isna().sum() == 0
        assert "median" in desc

    def test_mean_rejects_text_column(self, dirty_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="numeric"):
            impute(dirty_df, "grup", "mean")

    def test_drop_rows(self, dirty_df: pd.DataFrame) -> None:
        out, _ = impute(dirty_df, "value", "drop_rows")
        assert len(out) == len(dirty_df) - 1

    def test_input_is_not_mutated(self, dirty_df: pd.DataFrame) -> None:
        before = dirty_df["value"].isna().sum()
        impute(dirty_df, "value", "median")
        assert dirty_df["value"].isna().sum() == before


class TestOutliers:
    def test_iqr_clip_keeps_row_count(self, dirty_df: pd.DataFrame) -> None:
        out, _ = handle_outliers(dirty_df, "value", "iqr", "clip")
        assert len(out) == len(dirty_df)
        assert out["value"].max() < 1000

    def test_iqr_drop_removes_row(self, dirty_df: pd.DataFrame) -> None:
        out, _ = handle_outliers(dirty_df, "value", "iqr", "drop")
        assert len(out) < len(dirty_df)


class TestCast:
    def test_numeric_cast_reports_losses(self) -> None:
        df = pd.DataFrame({"a": ["1", "2", "abc"]})
        out, desc = cast_column(df, "a", "numeric")
        assert out["a"].isna().sum() == 1
        assert "became NaN" in desc


class TestFilter:
    def test_greater_than(self, dirty_df: pd.DataFrame) -> None:
        out, desc = apply_filter(dirty_df, "value", ">", 3)
        assert len(out) == 3
        assert "kept 3" in desc

    def test_contains_is_null_safe(self, dirty_df: pd.DataFrame) -> None:
        out, _ = apply_filter(dirty_df, "grup", "contains", "a")
        assert len(out) == 3


class TestHistory:
    def test_undo_redo_round_trip(self, dirty_df: pd.DataFrame) -> None:
        h = History(dirty_df)
        new, desc = impute(dirty_df, "value", "median")
        h.commit(new, "impute", desc)

        assert h.current["value"].isna().sum() == 0
        assert h.can_undo and not h.can_redo

        h.undo()
        assert h.current["value"].isna().sum() == 1
        assert h.can_redo

        h.redo()
        assert h.current["value"].isna().sum() == 0

    def test_new_commit_discards_redo_branch(self, dirty_df: pd.DataFrame) -> None:
        h = History(dirty_df)
        h.commit(dirty_df.assign(value=0), "a", "a")
        h.undo()
        h.commit(dirty_df.assign(value=9), "b", "b")
        assert not h.can_redo
        assert (h.current["value"] == 9).all()

    def test_depth_is_bounded(self, dirty_df: pd.DataFrame) -> None:
        h = History(dirty_df, depth=3)
        for i in range(10):
            h.commit(dirty_df.assign(value=i), f"op{i}", "x")
        assert len(h._states) <= 4  # depth + current

    def test_operations_record_actor(self, dirty_df: pd.DataFrame) -> None:
        h = History(dirty_df)
        h.commit(dirty_df.head(3), "filter", "kept 3", actor="agent")
        row = h.table().iloc[0]
        assert row["actor"] == "agent"
        assert row["rows"] == "6 -> 3"
