"""AgentSession: the single mutable object the agent is allowed to touch.

The LLM never receives the DataFrame. It only sees short text observations,
and every mutation goes through History with actor="agent", so the human can
audit and undo each step from the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from atml.modeling import TrainResult
from atml.processing import History


@dataclass
class AgentSession:
    history: History
    train_result: TrainResult | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> AgentSession:
        return cls(History(df))

    @property
    def df(self) -> pd.DataFrame:
        return self.history.current

    def commit(self, new_df: pd.DataFrame, name: str, description: str) -> str:
        op = self.history.commit(new_df, name, description, actor="agent")
        self.trace.append(op.as_row())
        return description
