from __future__ import annotations

import streamlit as st
from app.state import commit, history_sidebar, require_data

from atml.processing import (
    add_calculated_column,
    apply_filter,
    cast_column,
    handle_outliers,
    impute,
)
from atml.validation import FormulaError

df = require_data()
st.title("Clean")
st.caption("Every operation is committed to the history and can be undone from the sidebar.")

missing = df.isna().sum()
if missing.any():
    st.warning(f"{int(missing.sum()):,} missing values across {int((missing > 0).sum())} columns")

t_missing, t_outlier, t_type, t_formula, t_filter = st.tabs(
    ["Missing values", "Outliers", "Types", "Calculated column", "Filter"]
)

with t_missing:
    cols = missing[missing > 0].index.tolist()
    if not cols:
        st.success("No missing values.")
    else:
        col = st.selectbox("Column", cols, format_func=lambda c: f"{c} ({missing[c]} missing)")
        strategy = st.selectbox(
            "Strategy", ["median", "mean", "mode", "constant", "ffill", "bfill", "drop_rows"]
        )
        value = st.text_input("Fill value") if strategy == "constant" else None
        if st.button("Apply", type="primary", key="imp"):
            try:
                new, desc = impute(df, col, strategy, value)
                commit(new, "impute", desc)
                st.success(desc)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

with t_outlier:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.info("No numeric columns.")
    else:
        col = st.selectbox("Column", num_cols, key="out_col")
        method = st.radio("Method", ["iqr", "zscore"], horizontal=True)
        action = st.radio("Action", ["clip", "drop"], horizontal=True)
        if st.button("Apply", type="primary", key="out"):
            new, desc = handle_outliers(df, col, method, action)
            commit(new, "outliers", desc)
            st.success(desc)
            st.rerun()

with t_type:
    col = st.selectbox("Column", df.columns.tolist(), key="cast_col")
    target = st.selectbox("Convert to", ["numeric", "datetime", "category", "string"])
    if st.button("Convert", type="primary", key="cast"):
        new, desc = cast_column(df, col, target)
        commit(new, "cast", desc)
        st.success(desc)
        st.rerun()

with t_formula:
    st.caption(
        "Formulas are parsed into an AST and restricted to arithmetic plus "
        "abs, log, log1p, exp, sqrt, round, min, max, where. `eval` is never called."
    )
    name = st.text_input("New column name")
    formula = st.text_input("Formula", placeholder="canlı_ağırlık_kg / yaş")
    if st.button("Add column", type="primary", key="calc") and name and formula:
        try:
            new, desc = add_calculated_column(df, name, formula)
            commit(new, "formula", desc)
            st.success(desc)
            st.rerun()
        except FormulaError as exc:
            st.error(f"Rejected: {exc}")

with t_filter:
    col = st.selectbox("Column", df.columns.tolist(), key="f_col")
    op = st.selectbox(
        "Operator", ["==", "!=", ">", ">=", "<", "<=", "contains", "is_null", "not_null"]
    )
    val = st.text_input("Value") if op not in ("is_null", "not_null") else None
    if st.button("Apply filter", type="primary", key="filt"):
        parsed: object = val
        try:
            parsed = float(val) if val is not None else None
        except (TypeError, ValueError):
            pass
        new, desc = apply_filter(df, col, op, parsed)
        commit(new, "filter", desc)
        st.success(desc)
        st.rerun()

st.subheader("Current data")
st.dataframe(df.head(200), use_container_width=True)
history_sidebar()
