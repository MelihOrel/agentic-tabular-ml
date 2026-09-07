from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st
from app.state import history_sidebar, require_data

from atml.io import load_file
from atml.modeling import AutoML, detect_task
from atml.modeling.automl import predict_frame
from atml.validation import MLGuard, Severity

df = require_data()
st.title("Model")

target = st.selectbox("Target column", df.columns.tolist())
task = detect_task(df[target])
st.caption(f"Detected task: **{task}**")

report = MLGuard().check(df, target)
for issue in report.issues:
    {Severity.ERROR: st.error, Severity.WARNING: st.warning, Severity.INFO: st.info}[
        issue.severity
    ](f"**{issue.column or 'dataset'}** — {issue.message}")

if not report.ok:
    st.stop()

if st.button("Train models", type="primary"):
    with st.spinner("Cross-validating..."):
        st.session_state.train_result = AutoML().fit(df, target)

result = st.session_state.get("train_result")
if result is None:
    st.stop()

st.subheader("Leaderboard")
board = pd.DataFrame(
    [
        {"model": e.model, f"cv_{e.metric}": round(e.cv_mean, 4), "cv_std": round(e.cv_std, 4)}
        for e in result.leaderboard
    ]
)
st.dataframe(board, hide_index=True, use_container_width=True)
st.caption(
    f"{result.n_train} training rows, {result.n_test} held-out test rows. "
    "Leaderboard scores are cross-validated on the training split only; "
    "the test scores below were computed once, on data no model saw."
)

cols = st.columns(len(result.test_scores))
for c, (k, v) in zip(cols, result.test_scores.items(), strict=True):
    c.metric(f"test {k}", f"{v:.4f}")

st.subheader("Feature importance")
st.caption("Permutation importance on the test set: the drop in score when a column is shuffled.")
imp = pd.Series(result.feature_importance).head(15).sort_values()
st.plotly_chart(
    px.bar(imp, orientation="h", labels={"value": "importance", "index": ""}),
    use_container_width=True,
)

st.subheader("Use the model")
t_single, t_batch, t_export = st.tabs(["Single prediction", "Batch file", "Export"])

with t_single:
    inputs = {}
    for col in result.features:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            inputs[col] = st.number_input(col, value=float(s.median()))
        else:
            inputs[col] = st.selectbox(col, s.dropna().unique().tolist(), key=f"in_{col}")
    if st.button("Predict", type="primary"):
        out = predict_frame(result.pipeline, result.summary(), pd.DataFrame([inputs]))
        st.metric("Prediction", str(out["prediction"].iloc[0]))
        if "confidence" in out:
            st.caption(f"Confidence: {out['confidence'].iloc[0]:.1%}")

with t_batch:
    up = st.file_uploader("File with the same feature columns", type=["csv", "xlsx"], key="batch")
    if up is not None:
        rows, _ = load_file(up.getvalue(), up.name)
        try:
            out = predict_frame(result.pipeline, result.summary(), rows)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.dataframe(out.head(200), use_container_width=True)
            st.download_button(
                "Download predictions",
                out.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv",
            )

with t_export:
    st.caption(
        "The exported file contains the full pipeline (preprocessing + model) and metadata, "
        "ready for `api/main.py`."
    )
    buf = io.BytesIO()
    import joblib

    joblib.dump({"pipeline": result.pipeline, "meta": result.summary()}, buf)
    st.download_button("Download champion.joblib", buf.getvalue(), "champion.joblib")

history_sidebar()
