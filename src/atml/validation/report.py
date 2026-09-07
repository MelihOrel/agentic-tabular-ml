from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Issue:
    severity: Severity
    message: str
    column: str | None = None


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)

    def add(self, severity: Severity, message: str, column: str | None = None) -> None:
        self.issues.append(Issue(severity, message, column))

    @property
    def ok(self) -> bool:
        return not any(i.severity is Severity.ERROR for i in self.issues)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    def summary(self) -> str:
        if not self.issues:
            return "No issues."
        return "\n".join(
            f"[{i.severity.value}] {i.column or '-'}: {i.message}" for i in self.issues
        )
