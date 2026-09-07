"""Agent tools are tested WITHOUT an LLM.

The point of these tests is that the guard rails hold no matter what the model
asks for: bad formulas are rejected, unusable targets never reach training,
and every mutation lands in the history as actor="agent" so a human can undo it.
"""

from __future__ import annotations

import json

import pandas as pd

from atml.agent import AgentSession
from atml.agent.tools import make_tool_functions


def test_inspect_reports_shape_and_missing(dirty_df: pd.DataFrame) -> None:
    tools = make_tool_functions(AgentSession.from_frame(dirty_df))
    info = json.loads(tools["inspect_data"]())
    assert info["rows"] == 6
    assert info["columns"]["value"]["missing"] == 1


def test_mutations_are_recorded_as_agent_actions(dirty_df: pd.DataFrame) -> None:
    session = AgentSession.from_frame(dirty_df)
    tools = make_tool_functions(session)
    tools["impute_column"]("value", "median")

    assert session.df["value"].isna().sum() == 0
    assert session.history.table().iloc[0]["actor"] == "agent"
    assert session.history.can_undo


def test_undo_tool_reverts(dirty_df: pd.DataFrame) -> None:
    session = AgentSession.from_frame(dirty_df)
    tools = make_tool_functions(session)
    tools["impute_column"]("value", "median")
    tools["undo_last"]()
    assert session.df["value"].isna().sum() == 1


def test_unsafe_formula_is_refused_not_raised(dirty_df: pd.DataFrame) -> None:
    """The agent must get a readable observation back, not a crashed executor."""
    session = AgentSession.from_frame(dirty_df)
    tools = make_tool_functions(session)
    msg = tools["add_column"]("hack", "__import__('os').system('ls')")
    assert msg.startswith("Formula rejected")
    assert "hack" not in session.df.columns
    assert not session.history.can_undo  # nothing was committed


def test_training_blocked_by_guard(dirty_df: pd.DataFrame) -> None:
    session = AgentSession.from_frame(dirty_df)  # only 6 rows
    tools = make_tool_functions(session)
    out = tools["train_model"]("grup")
    assert "refused by MLGuard" in out
    assert session.train_result is None


def test_training_succeeds_and_returns_leaderboard(clf_df: pd.DataFrame) -> None:
    session = AgentSession.from_frame(clf_df)
    tools = make_tool_functions(session)
    payload = json.loads(tools["train_model"]("target"))
    assert payload["task"] == "classification"
    assert len(payload["leaderboard"]) == 3
    assert session.train_result is not None
