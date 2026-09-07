from __future__ import annotations

from pathlib import Path

import streamlit as st
from app.state import get_df, history_sidebar, set_data

from atml.io import load_file
from atml.validation import sanitize_columns

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"

st.title("Load")
st.caption(
    "Encoding, delimiter and decimal separator are detected and reported, not silently guessed."
)

tab_upload, tab_sample = st.tabs(["Upload a file", "Use a sample"])

with tab_upload:
    uploaded = st.file_uploader(
        "CSV, TSV, TXT, XLSX or XLSM", type=["csv", "tsv", "txt", "xlsx", "xlsm"]
    )
    with st.expander("Override detection"):
        enc = st.text_input("Encoding", placeholder="auto")
        delim = st.selectbox("Delimiter", ["auto", ",", ";", "\\t", "|"])
        header_row = st.number_input("Header row", min_value=0, value=0)
    if uploaded is not None and st.button("Load file", type="primary"):
        df, report = load_file(
            uploaded.getvalue(),
            uploaded.name,
            header_row=int(header_row),
            encoding=enc or None,
            delimiter={"auto": None, "\\t": "\t"}.get(delim, delim),
        )
        df, schema_report = sanitize_columns(df)
        set_data(df, report)
        for issue in schema_report.issues:
            st.write(f"- {issue.message}")
        st.rerun()

with tab_sample:
    files = sorted(SAMPLES.glob("*")) if SAMPLES.exists() else []
    if not files:
        st.warning("Run `python scripts/make_sample_data.py` to generate the demo files.")
    else:
        choice = st.selectbox("Sample file", [f.name for f in files])
        st.caption(
            "`keci_surusu_cp1254.csv` is deliberately awkward: CP1254 encoding, "
            "`;` delimiter, `,` decimals and Turkish column names."
        )
        if st.button("Load sample", type="primary"):
            path = SAMPLES / choice
            df, report = load_file(path.read_bytes(), path.name)
            df, _ = sanitize_columns(df)
            set_data(df, report)
            st.rerun()

df = get_df()
if df is not None:
    report = st.session_state.load_report
    if report is not None:
        c = st.columns(4)
        c[0].metric("Rows", f"{report.rows:,}")
        c[1].metric("Columns", report.columns)
        c[2].metric("Encoding", report.encoding or "-")
        c[3].metric("Delimiter", repr(report.delimiter) if report.delimiter else "-")
        if report.numeric_conversions:
            st.success(
                "Converted to numeric from European format: "
                + ", ".join(report.numeric_conversions)
            )
        for note in report.notes:
            st.info(note)

    st.subheader("Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Column types")
    st.dataframe(
        df.dtypes.astype(str)
        .rename("dtype")
        .to_frame()
        .assign(missing=df.isna().sum(), unique=df.nunique()),
        use_container_width=True,
    )

history_sidebar()
