from __future__ import annotations

import plotly.express as px
import streamlit as st
from app.state import history_sidebar, require_data

from atml.dtypes import text_columns
from atml.stats import (
    anova,
    chi_square,
    describe,
    isolation_forest,
    kmeans_elbow,
    normality,
    pca_2d,
    ttest,
)

df = require_data()
st.title("Explore")

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = text_columns(df)

t_summary, t_dist, t_tests, t_unsup = st.tabs(
    ["Summary", "Distributions", "Hypothesis tests", "Unsupervised"]
)

with t_summary:
    if num_cols:
        st.dataframe(describe(df).round(3), use_container_width=True)
    if len(num_cols) >= 2:
        st.subheader("Correlation")
        st.plotly_chart(
            px.imshow(
                df[num_cols].corr().round(2),
                text_auto=True,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                aspect="auto",
            ),
            use_container_width=True,
        )

with t_dist:
    if num_cols:
        col = st.selectbox("Column", num_cols)
        colour = st.selectbox("Colour by", ["none", *cat_cols])
        st.plotly_chart(
            px.histogram(df, x=col, color=None if colour == "none" else colour, marginal="box"),
            use_container_width=True,
        )
        st.json(normality(df[col]))

with t_tests:
    test = st.selectbox("Test", ["Welch t-test", "One-way ANOVA", "Chi-square"])
    if test == "Welch t-test" and num_cols and cat_cols:
        v = st.selectbox("Value", num_cols, key="t_v")
        g = st.selectbox("Group (2 levels)", cat_cols, key="t_g")
        if st.button("Run", type="primary", key="t_run"):
            st.json(ttest(df, v, g))
    elif test == "One-way ANOVA" and num_cols and cat_cols:
        v = st.selectbox("Value", num_cols, key="a_v")
        g = st.selectbox("Group", cat_cols, key="a_g")
        if st.button("Run", type="primary", key="a_run"):
            st.json(anova(df, v, g))
    elif test == "Chi-square" and len(cat_cols) >= 2:
        a = st.selectbox("Column A", cat_cols, key="c_a")
        b = st.selectbox("Column B", cat_cols, key="c_b")
        if st.button("Run", type="primary", key="c_run"):
            st.json(chi_square(df, a, b))
    else:
        st.info("This test needs numeric and/or categorical columns that the data does not have.")

with t_unsup:
    if len(num_cols) < 2:
        st.info("Needs at least two numeric columns.")
    else:
        method = st.radio("Method", ["K-Means", "Isolation Forest", "PCA"], horizontal=True)
        if method == "K-Means":
            res = kmeans_elbow(df[num_cols])
            st.caption(f"Chosen k = {res['k']} (elbow on inertia)")
            st.plotly_chart(
                px.line(
                    x=list(res["inertia_by_k"]),
                    y=list(res["inertia_by_k"].values()),
                    labels={"x": "k", "y": "inertia"},
                    markers=True,
                ),
                use_container_width=True,
            )
            coords = pca_2d(df[num_cols])["coords"].join(res["labels"].astype(str))
            st.plotly_chart(
                px.scatter(coords, x="PC1", y="PC2", color="cluster"), use_container_width=True
            )
        elif method == "Isolation Forest":
            res = isolation_forest(df[num_cols])
            st.metric("Anomalies", res["n_anomalies"])
            st.dataframe(
                df.loc[res["flags"][res["flags"]].index].head(50), use_container_width=True
            )
        else:
            res = pca_2d(df[num_cols])
            st.caption(f"Explained variance: {res['explained_variance']}")
            st.plotly_chart(px.scatter(res["coords"], x="PC1", y="PC2"), use_container_width=True)

history_sidebar()
