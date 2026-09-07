"""Session state helpers. All pages read and write the data through here."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from atml.agent import AgentSession
from atml.processing import History


def init_state() -> None:
    st.session_state.setdefault("session", None)
    st.session_state.setdefault("load_report", None)
    st.session_state.setdefault("train_result", None)


def set_data(df: pd.DataFrame, load_report: Any | None = None) -> None:
    st.session_state.session = AgentSession(History(df))
    st.session_state.load_report = load_report
    st.session_state.train_result = None


def get_session() -> AgentSession | None:
    return st.session_state.get("session")


def get_df() -> pd.DataFrame | None:
    s = get_session()
    return s.df if s else None


def commit(new_df: pd.DataFrame, name: str, description: str) -> None:
    st.session_state.session.history.commit(new_df, name, description, actor="user")


def require_data() -> pd.DataFrame:
    """Stop the page with a friendly message when nothing is loaded yet."""
    df = get_df()
    if df is None:
        st.info("Load a dataset on the **Load** page first.")
        st.stop()
    return df


def history_sidebar() -> None:
    """Undo/redo controls and the operation log, shared by every page."""
    session = get_session()
    if session is None:
        return
    hist = session.history
    with st.sidebar:
        st.caption(f"{len(hist.current):,} rows x {hist.current.shape[1]} columns")
        c1, c2 = st.columns(2)
        if c1.button("Undo", disabled=not hist.can_undo, use_container_width=True):
            hist.undo()
            st.rerun()
        if c2.button("Redo", disabled=not hist.can_redo, use_container_width=True):
            hist.redo()
            st.rerun()
        table = hist.table()
        if not table.empty:
            with st.expander(f"History ({len(table)})"):
                st.dataframe(table, hide_index=True, use_container_width=True)
