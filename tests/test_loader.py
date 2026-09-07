"""Loader tests use adversarial files on purpose: BOM, CP1254, mixed delimiters,
European decimals and quoted commas. These are the cases pandas' defaults get wrong.
"""

from __future__ import annotations

import pandas as pd
import pytest

from atml.io import detect_decimal, detect_delimiter, detect_encoding, load_file

TR_HEADER = "ad;yaş;ağırlık\n"
TR_ROWS = "Şükrü;34;1.250,50\nÇiğdem;28;980,25\nİsmail;41;1.010,00\n"


@pytest.mark.parametrize(
    ("encoding", "expected"),
    [("utf-8", "utf-8"), ("cp1254", "cp1254"), ("iso-8859-9", "cp1254")],
)
def test_detect_encoding_turkish(encoding: str, expected: str) -> None:
    raw = (TR_HEADER + TR_ROWS).encode(encoding)
    # cp1254 and iso-8859-9 share byte values for Turkish letters; either decodes cleanly.
    assert detect_encoding(raw) == expected


def test_detect_encoding_bom() -> None:
    raw = (TR_HEADER + TR_ROWS).encode("utf-8-sig")
    assert detect_encoding(raw) == "utf-8-sig"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a,b,c\n1,2,3\n4,5,6\n", ","),
        ("a;b;c\n1;2;3\n4;5;6\n", ";"),
        ("a\tb\tc\n1\t2\t3\n4\t5\t6\n", "\t"),
        ("a|b|c\n1|2|3\n4|5|6\n", "|"),
        # A comma inside quoted text must not beat the real ';' delimiter.
        ('ad;not\n"Ali, Veli";5\n"Ayşe, Fatma";7\n', ";"),
    ],
)
def test_detect_delimiter(text: str, expected: str) -> None:
    assert detect_delimiter(text) == expected


def test_detect_decimal_european() -> None:
    assert detect_decimal(TR_HEADER + TR_ROWS, ";") == ","


def test_detect_decimal_us() -> None:
    text = "a,b\n1.5,2000.25\n3.25,1000.00\n"
    assert detect_decimal(text, ",") == "."


def test_load_turkish_csv_end_to_end() -> None:
    raw = (TR_HEADER + TR_ROWS).encode("cp1254")
    df, report = load_file(raw, "kayit.csv")

    assert report.encoding == "cp1254"
    assert report.delimiter == ";"
    assert report.decimal == ","
    assert list(df.columns) == ["ad", "yaş", "ağırlık"]
    assert df["ad"].tolist() == ["Şükrü", "Çiğdem", "İsmail"]  # no mojibake
    assert pd.api.types.is_numeric_dtype(df["ağırlık"])
    assert df["ağırlık"].iloc[0] == pytest.approx(1250.50)


def test_load_bundled_sample_files() -> None:
    """The shipped demo files must load correctly, otherwise the README lies."""
    from pathlib import Path

    samples = Path(__file__).resolve().parents[1] / "data" / "samples"
    if not (samples / "keci_surusu_cp1254.csv").exists():
        pytest.skip("run scripts/make_sample_data.py first")

    df, report = load_file((samples / "keci_surusu_cp1254.csv").read_bytes(), "keci.csv")
    assert report.delimiter == ";"
    assert "canlı_ağırlık_kg" in df.columns
    assert pd.api.types.is_numeric_dtype(df["canlı_ağırlık_kg"])


def test_load_excel() -> None:
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "data" / "samples" / "goat_herd.xlsx"
    if not p.exists():
        pytest.skip("run scripts/make_sample_data.py first")
    df, report = load_file(p.read_bytes(), "goat_herd.xlsx")
    assert report.source_kind == "excel"
    assert report.rows == len(df) > 0
