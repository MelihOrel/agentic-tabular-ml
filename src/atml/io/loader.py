"""Encoding, delimiter and decimal aware loading for CSV / Excel.

Why this exists: Turkish exports from Excel are typically CP1254 with ';' as the
delimiter and ',' as the decimal mark. pandas' defaults silently produce mojibake
('Åž' instead of 'Ş') and object columns full of '1.250,50' strings. Every
decision made here is written into a LoadReport so the UI and the agent can show it.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from atml.config import load_config
from atml.dtypes import text_columns

_EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xls")
_CSV_SUFFIXES = (".csv", ".tsv", ".txt")
_TURKISH_CHARS = "çğıöşüÇĞİÖŞÜ"
_MOJIBAKE_MARKERS = ("Ã", "Å", "Ä±", "�")


@dataclass
class LoadReport:
    """Everything decided during a load, so nothing happens silently."""

    encoding: str | None = None
    delimiter: str | None = None
    decimal: str = "."
    source_kind: str = "csv"
    sheet: str | None = None
    rows: int = 0
    columns: int = 0
    numeric_conversions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# --------------------------------------------------------------------------- detection


def detect_encoding(raw: bytes, candidates: list[str] | None = None) -> str:
    """Try candidate encodings in order; pick the first that decodes without mojibake.

    Preference order matters: utf-8-sig catches BOM files, utf-8 is strict and
    fails fast on CP1254 bytes, and CP1254 is preferred over latin-1 because
    latin-1 decodes *anything* but maps Turkish letters to the wrong glyphs.
    """
    cfg = load_config()["loader"]
    candidates = candidates or cfg["encodings"]
    sample = raw[: cfg["sample_bytes"]]

    # A BOM is decisive; without one, never report utf-8-sig (it would decode plain
    # UTF-8 fine but misleads anyone reading the report).
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    candidates = [e for e in candidates if e != "utf-8-sig"]

    for enc in candidates:
        try:
            text = sample.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        if any(marker in text for marker in _MOJIBAKE_MARKERS):
            continue
        return enc
    return "latin-1"


def detect_delimiter(sample_text: str, candidates: list[str] | None = None) -> str:
    """Score each delimiter by how consistent the field count is across lines."""
    candidates = candidates or load_config()["loader"]["delimiters"]
    lines = [ln for ln in sample_text.splitlines()[:50] if ln.strip()]
    if len(lines) < 2:
        return ","

    best, best_score = ",", -1.0
    for delim in candidates:
        counts = [ln.count(delim) for ln in lines]
        if max(counts) == 0:
            continue
        mode, freq = Counter(counts).most_common(1)[0]
        consistency = freq / len(counts)
        score = consistency * (1 + mode)  # consistent AND actually splits
        if score > best_score:
            best, best_score = delim, score
    return best


_EU_NUMBER = re.compile(r"^-?\d{1,3}(\.\d{3})*(,\d+)?$|^-?\d+,\d+$")
_US_NUMBER = re.compile(r"^-?\d{1,3}(,\d{3})*(\.\d+)?$|^-?\d+\.\d+$")


def detect_decimal(sample_text: str, delimiter: str) -> str:
    """Return ',' when European style numbers dominate the sample, else '.'."""
    eu = us = 0
    for line in sample_text.splitlines()[1:200]:
        for tok in line.split(delimiter):
            tok = tok.strip().strip('"')
            if not tok or not any(ch.isdigit() for ch in tok):
                continue
            if "," in tok and _EU_NUMBER.match(tok):
                eu += 1
            elif "." in tok and _US_NUMBER.match(tok):
                us += 1
    if delimiter == "," and eu:
        # A comma delimited file cannot also use comma decimals unless quoted.
        return "," if eu > us * 3 else "."
    return "," if eu > us else "."


# --------------------------------------------------------------------------- loading


def _coerce_european_numbers(df: pd.DataFrame, report: LoadReport) -> pd.DataFrame:
    """Convert object columns like '1.250,50' into floats when nearly all values parse."""
    for col in text_columns(df):
        s = df[col].dropna().astype(str).str.strip()
        if s.empty:
            continue
        looks_numeric = s.str.match(_EU_NUMBER.pattern).mean()
        if looks_numeric >= 0.95:
            converted = pd.to_numeric(
                s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            if converted.notna().mean() >= 0.95:
                df[col] = pd.to_numeric(
                    df[col]
                    .astype(str)
                    .str.replace(".", "", regex=False)
                    .str.replace(",", ".", regex=False),
                    errors="coerce",
                )
                report.numeric_conversions.append(col)
    return df


def load_file(
    raw: bytes,
    filename: str,
    *,
    sheet_name: str | int | None = 0,
    header_row: int = 0,
    delimiter: str | None = None,
    encoding: str | None = None,
) -> tuple[pd.DataFrame, LoadReport]:
    """Load CSV/TSV/TXT/XLSX bytes into a DataFrame with a full LoadReport.

    Manual overrides (delimiter, encoding) win over detection so the UI can expose them.
    """
    cfg = load_config()["loader"]
    name = filename.lower()
    report = LoadReport()

    if name.endswith(_EXCEL_SUFFIXES):
        report.source_kind = "excel"
        df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name, header=header_row)
        report.sheet = str(sheet_name)
    elif name.endswith(_CSV_SUFFIXES) or not any(name.endswith(s) for s in _EXCEL_SUFFIXES):
        report.source_kind = "csv"
        report.encoding = encoding or detect_encoding(raw)
        text = raw.decode(report.encoding, errors="replace")
        report.delimiter = delimiter or detect_delimiter(text[: cfg["sample_bytes"]])
        report.decimal = detect_decimal(text[: cfg["sample_bytes"]], report.delimiter)
        df = pd.read_csv(
            io.StringIO(text),
            sep=report.delimiter,
            decimal=report.decimal,
            thousands="." if report.decimal == "," else None,
            header=header_row,
            nrows=cfg["max_rows"],
            engine="python",
        )
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported file type: {filename}")

    df = _coerce_european_numbers(df, report)
    report.rows, report.columns = df.shape
    if any(ch in "".join(map(str, df.columns)) for ch in _TURKISH_CHARS):
        report.notes.append("Turkish characters detected in headers and decoded cleanly.")
    return df, report
