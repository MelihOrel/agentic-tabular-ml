"""Undo/redo history for DataFrame operations.

Both the Streamlit UI and the ReAct agent push operations through here, so every
step the agent takes can be inspected and reverted by a human.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from atml.config import load_config


@dataclass
class Operation:
    name: str
    description: str
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    actor: str = "user"  # "user" or "agent"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def as_row(self) -> dict:
        return {
            "time": self.timestamp,
            "actor": self.actor,
            "operation": self.name,
            "description": self.description,
            "rows": f"{self.rows_before} -> {self.rows_after}",
            "columns": f"{self.cols_before} -> {self.cols_after}",
        }


class History:
    """Snapshot-based history. Fine for in-memory tabular data of modest size."""

    def __init__(self, initial: pd.DataFrame, depth: int | None = None):
        self.depth = depth or load_config()["cleaning"]["history_depth"]
        self._states: list[pd.DataFrame] = [initial.copy()]
        self._ops: list[Operation] = []
        self._cursor = 0  # index into _states

    # ----------------------------------------------------------------- properties
    @property
    def current(self) -> pd.DataFrame:
        return self._states[self._cursor]

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._cursor < len(self._states) - 1

    @property
    def operations(self) -> list[Operation]:
        return self._ops[: self._cursor]

    # ------------------------------------------------------------------ mutators
    def commit(
        self, new_df: pd.DataFrame, name: str, description: str, actor: str = "user"
    ) -> Operation:
        old = self.current
        op = Operation(
            name, description, len(old), len(new_df), old.shape[1], new_df.shape[1], actor
        )
        # Drop any redo branch, then append.
        self._states = self._states[: self._cursor + 1] + [new_df.copy()]
        self._ops = self._ops[: self._cursor] + [op]
        if len(self._states) > self.depth + 1:
            self._states.pop(0)
            self._ops.pop(0)
        self._cursor = len(self._states) - 1
        return op

    def undo(self) -> Operation | None:
        if not self.can_undo:
            return None
        self._cursor -= 1
        return self._ops[self._cursor]

    def redo(self) -> Operation | None:
        if not self.can_redo:
            return None
        op = self._ops[self._cursor]
        self._cursor += 1
        return op

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([o.as_row() for o in self.operations])
