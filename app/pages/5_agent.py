from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from app.state import get_session, history_sidebar, require_data

from atml.agent.tools import make_tool_functions

require_data()
session = get_session()

st.title("Agent")
st.caption(
    "The agent works only through the tools below. It never sees the raw data, "
    "every change it makes is recorded in the history as an agent action, and you can undo it."
)

with st.expander("Available tools", expanded=False):
    for name, fn in make_tool_functions(session).items():
        st.markdown(f"**`{name}`** — {(fn.__doc__ or '').strip().splitlines()[0]}")

has_key = bool(os.getenv("OPENAI_API_KEY"))
if not has_key:
    st.warning(
        "No `OPENAI_API_KEY` found, so the LLM loop is disabled. Everything else in "
        "this app works without it. Copy `.env.example` to `.env` to enable the agent."
    )

task = st.text_area(
    "Task",
    placeholder=(
        "Clean the missing values sensibly, then train a model "
        "predicting vücut_kondisyon_skoru."
    ),
    height=90,
)

if st.button("Run agent", type="primary", disabled=not (has_key and task)):
    try:
        from atml.agent.react_agent import run_agent
    except ImportError:
        st.error("LangChain is not installed. Run: pip install -e '.[agent]'")
    else:
        with st.spinner("Agent working..."):
            result = run_agent(session, task)
        st.subheader("Steps")
        for i, step in enumerate(result["steps"], 1):
            with st.expander(f"{i}. {step['tool']}", expanded=False):
                st.json(step["input"])
                st.code(step["observation"], language="json")
        st.subheader("Answer")
        st.markdown(result["output"])
        st.rerun()

trace = session.trace
if trace:
    st.subheader("What the agent changed")
    st.dataframe(pd.DataFrame(trace), hide_index=True, use_container_width=True)

history_sidebar()
