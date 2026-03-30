"""Execution context for program runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepResult:
    """Result of executing a single directive."""

    agent: str
    action: str
    output: Any = None
    success: bool = True
    error: str | None = None
    duration_ms: float = 0.0

    def __repr__(self) -> str:
        status = "OK" if self.success else f"ERR({self.error})"
        return f"StepResult(@{self.agent} {self.action} → {status})"


@dataclass
class ExecutionContext:
    """Mutable context passed through execution.

    Tracks step results, blackboard state, and metadata.
    """

    # Results from previous steps (for $prev resolution)
    step_results: list[StepResult] = field(default_factory=list)

    # Blackboard state (namespace:key → value)
    blackboard: dict[str, Any] = field(default_factory=dict)

    # Variables set by the program
    variables: dict[str, Any] = field(default_factory=dict)

    # Execution metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Sender agent (who initiated this execution)
    sender: str | None = None

    @property
    def prev(self) -> Any:
        """Get the most recent step result's output."""
        if not self.step_results:
            return None
        return self.step_results[-1].output

    def get_prev(self, index: int) -> Any:
        """Get a specific step result's output by index."""
        if 0 <= index < len(self.step_results):
            return self.step_results[index].output
        return None

    def resolve_bb(self, namespace: str, key: str, version: int | None = None) -> Any:
        """Resolve a blackboard pointer."""
        pointer = f"{namespace}:{key}"
        return self.blackboard.get(pointer)

    def push_result(self, result: StepResult) -> None:
        self.step_results.append(result)
