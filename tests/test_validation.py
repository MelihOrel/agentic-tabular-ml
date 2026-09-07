from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atml.validation import FormulaError, MLGuard, Severity, evaluate_formula, sanitize_columns


class TestSchema:
    def test_duplicate_and_unnamed_columns(self) -> None:
        df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "Unnamed: 2"])
        out, report = sanitize_columns(df)
        assert len(set(out.columns)) == 3
        assert "column_2" in out.columns
        assert report.warnings

    def test_strips_html_but_keeps_turkish(self) -> None:
        df = pd.DataFrame([[1, 2]], columns=["<script>x</script>", " ağırlık "])
        out, _ = sanitize_columns(df)
        assert "<" not in "".join(out.columns)
        assert "ağırlık" in out.columns


class TestFormula:
    def test_arithmetic(self) -> None:
        df = pd.DataFrame({"a": [10.0, 20.0], "b": [2.0, 4.0]})
        assert evaluate_formula(df, "a / b * 2").tolist() == [10.0, 10.0]

    def test_whitelisted_function(self) -> None:
        df = pd.DataFrame({"a": [1.0, 100.0]})
        assert np.isclose(evaluate_formula(df, "sqrt(a)").iloc[1], 10.0)
        assert np.isclose(evaluate_formula(df, "abs(0 - a)").iloc[0], 1.0)

    @pytest.mark.parametrize(
        "formula",
        [
            "__import__('os').system('ls')",
            "open('/etc/passwd').read()",
            "a.__class__.__mro__",
            "[x for x in range(10)]",
            "lambda: 1",
            "eval('1+1')",
        ],
    )
    def test_rejects_dangerous_expressions(self, formula: str) -> None:
        df = pd.DataFrame({"a": [1.0]})
        with pytest.raises(FormulaError):
            evaluate_formula(df, formula)

    def test_rejects_unknown_column(self) -> None:
        with pytest.raises(FormulaError, match="Unknown column"):
            evaluate_formula(pd.DataFrame({"a": [1.0]}), "b + 1")


class TestMLGuard:
    def test_accepts_reasonable_data(self, clf_df: pd.DataFrame) -> None:
        assert MLGuard().check(clf_df, "target").ok

    def test_rejects_single_class(self, clf_df: pd.DataFrame) -> None:
        df = clf_df.assign(target=1)
        report = MLGuard().check(df, "target")
        assert not report.ok
        assert "single value" in report.summary()

    def test_rejects_datetime_target(self) -> None:
        df = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=50), "x": range(50)})
        assert not MLGuard().check(df, "d").ok

    def test_rejects_tiny_dataset(self, clf_df: pd.DataFrame) -> None:
        assert not MLGuard().check(clf_df.head(5), "target").ok

    def test_warns_on_identifier_and_constant(self, clf_df: pd.DataFrame) -> None:
        df = clf_df.assign(
            row_id=[f"id{i}" for i in range(len(clf_df))],  # unique text -> identifier
            constant=1,
        )
        messages = [i.message for i in MLGuard().check(df, "target").issues]
        assert any("identifier" in m for m in messages)
        assert any("Zero-variance" in m for m in messages)

    def test_flags_target_leakage(self, reg_df: pd.DataFrame) -> None:
        df = reg_df.assign(leak=reg_df["price"])
        issues = MLGuard().check(df, "price").issues
        assert any(i.severity is Severity.WARNING and "leakage" in i.message for i in issues)


class TestPandas3Compatibility:
    """Regression tests for the pandas 3 `str` dtype.

    `series.dtype == object` and `select_dtypes(include="object")` stop matching
    text columns on pandas 3, which silently disabled the identifier warning and
    the mixed-type cast. atml.dtypes.is_textlike is the version-safe replacement.
    """

    def test_is_textlike_matches_string_and_object(self) -> None:
        from atml.dtypes import is_textlike

        assert is_textlike(pd.Series(["a", "b"]))
        assert is_textlike(pd.Series(["a", "b"], dtype="string"))
        assert is_textlike(pd.Series(["a", "b"], dtype="category"))
        assert not is_textlike(pd.Series([1, 2]))
        assert not is_textlike(pd.Series(pd.date_range("2024-01-01", periods=2)))

    def test_identifier_warning_fires_on_string_dtype(self, clf_df: pd.DataFrame) -> None:
        df = clf_df.assign(row_id=pd.Series([f"id{i}" for i in range(len(clf_df))], dtype="string"))
        messages = [i.message for i in MLGuard().check(df, "target").issues]
        assert any("identifier" in m for m in messages)
