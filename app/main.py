"""Streamlit entry point.

    streamlit run app/main.py

Pages live in app/pages/ and are registered here with st.navigation so the app
stays a thin UI layer: every page calls into atml.* and renders the result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Streamlit puts the entrypoint's folder on sys.path, not the repo root, so the
# `app.*` and `atml.*` imports below work no matter where `streamlit run` is called from.
ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.state import init_state  # noqa: E402

st.set_page_config(page_title="atml", page_icon="◧", layout="wide")


def main() -> None:
    init_state()
    pages = [
        st.Page("pages/1_load.py", title="Load", icon=":material/upload_file:"),
        st.Page("pages/2_clean.py", title="Clean", icon=":material/cleaning_services:"),
        st.Page("pages/3_explore.py", title="Explore", icon=":material/insights:"),
        st.Page("pages/4_model.py", title="Model", icon=":material/model_training:"),
        st.Page("pages/5_agent.py", title="Agent", icon=":material/smart_toy:"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
