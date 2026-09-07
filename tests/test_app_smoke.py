"""Streamlit pages are smoke-tested with AppTest so a broken page fails CI,
not the person who opens the app."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from atml.agent import AgentSession
from atml.processing import History

APP = Path(__file__).resolve().parents[1] / "app" / "main.py"
PAGES = sorted((APP.parent / "pages").glob("*.py"))


def _seeded(page: Path, df: pd.DataFrame) -> AppTest:
    at = AppTest.from_file(str(page), default_timeout=90)
    at.session_state["session"] = AgentSession(History(df))
    at.session_state["load_report"] = None
    at.session_state["train_result"] = None
    return at


def test_main_app_runs() -> None:
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_pages_render_with_data(page: Path, clf_df: pd.DataFrame) -> None:
    at = _seeded(page, clf_df).run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_pages_render_without_data(page: Path) -> None:
    """Pages must show a friendly prompt, not a traceback, before anything is loaded."""
    at = AppTest.from_file(str(page), default_timeout=60)
    at.session_state["session"] = None
    at.session_state["load_report"] = None
    at.session_state["train_result"] = None
    at.run()
    assert not at.exception
