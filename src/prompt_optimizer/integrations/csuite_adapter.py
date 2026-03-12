"""C-Suite execution adapter.

Bridges program execution to C-Suite's agent communication layer.
"""

from __future__ import annotations

from typing import Any, Protocol

from prompt_optimizer.runtime.context import ExecutionContext
from prompt_optimizer.runtime.executor import AgentAdapter


class CSuiteAgentProtocol(Protocol):
    """Minimal interface a C-Suite agent must expose."""

    code: str

    async def execute(self, task: Any) -> Any: ...
    async def reason(self, prompt: str, context: dict | None = None) -> str: ...


class CommunicatorProtocol(Protocol):
    """Minimal interface for C-Suite's communicator."""

    async def send_task(self, agent_code: str, task: Any, timeout: float) -> Any: ...
    async def query_agent(self, agent_code: str, query: str, context: dict | None = None) -> str: ...
    def get_agent(self, agent_code: str) -> Any: ...


class CSuiteExecutionAdapter(AgentAdapter):
    """Adapter that routes directives to C-Suite executives.

    Usage:
        from prompt_optimizer.grammar import Lexer, Parser
        from prompt_optimizer.runtime import Executor
        from prompt_optimizer.integrations.csuite_adapter import CSuiteExecutionAdapter

        adapter = CSuiteExecutionAdapter(communicator=cos.communicator)
        executor = Executor(adapter)

        tokens = Lexer().tokenize("@CFO ANALYZE revenue {period=Q1} -> summary !urgent")
        node = Parser(tokens).parse()
        result = await executor.execute(node)
    """

    # Maps actions to how they're dispatched
    QUERY_ACTIONS = frozenset({"ANALYZE", "ASSESS", "EVALUATE", "FORECAST", "REVIEW", "SUMMARIZE"})
    TASK_ACTIONS = frozenset({"GENERATE", "CREATE", "UPDATE", "DELETE", "EXECUTE", "OPTIMIZE", "PLAN"})

    def __init__(self, communicator: CommunicatorProtocol, timeout: float = 30.0) -> None:
        self._comm = communicator
        self._timeout = timeout

    async def execute_directive(
        self,
        agent: str,
        action: str,
        target: Any,
        params: dict[str, Any],
        constraints: list[str],
        context: ExecutionContext,
    ) -> Any:
        description = self._build_description(action, target, params, constraints)

        action_upper = action.upper()
        if action_upper in self.QUERY_ACTIONS:
            return await self._comm.query_agent(
                agent_code=agent,
                query=description,
                context={"params": params, "constraints": constraints, "ail_context": True},
            )
        else:
            task = _SimpleTask(
                description=description,
                task_type=action_upper,
                context={"params": params, "constraints": constraints},
            )
            return await self._comm.send_task(
                agent_code=agent,
                task=task,
                timeout=self._timeout,
            )

    async def evaluate_condition(self, condition: str, context: ExecutionContext) -> bool:
        """Evaluate by checking agent fields or simple comparisons."""
        import re
        m = re.match(
            r"@([A-Z]+)\.(\w+)\s*(>|<|>=|<=|==|!=)\s*(\S+)",
            condition.strip(),
        )
        if m:
            agent, field, op, raw_val = m.groups()
            result = await self._comm.query_agent(
                agent_code=agent,
                query=f"What is the current value of {field}?",
                context={"field": field, "ail_condition": True},
            )
            try:
                actual = self._parse_number(str(result))
                expected = self._parse_number(raw_val)
                return self._compare(actual, op, expected)
            except (ValueError, TypeError):
                return str(result).strip().lower() == raw_val.strip().lower()

        # Fallback: truthy check on prev result
        return bool(context.prev)

    def _build_description(self, action: str, target: Any, params: dict, constraints: list[str]) -> str:
        parts = [action]
        if target:
            parts.append(str(target))
        if params:
            kvs = ", ".join(f"{k}={v}" for k, v in params.items())
            parts.append(f"({kvs})")
        if constraints:
            parts.append(f"constraints: {', '.join(constraints)}")
        return " ".join(parts)

    @staticmethod
    def _parse_number(s: str) -> float:
        s = s.strip().replace(",", "")
        multipliers = {"k": 1_000, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
        if s and s[-1] in multipliers:
            return float(s[:-1]) * multipliers[s[-1]]
        return float(s)

    @staticmethod
    def _compare(a: float, op: str, b: float) -> bool:
        ops = {">": a > b, "<": a < b, ">=": a >= b, "<=": a <= b, "==": a == b, "!=": a != b}
        return ops.get(op, False)


class _SimpleTask:
    """Lightweight task object for C-Suite dispatch."""

    def __init__(self, description: str, task_type: str, context: dict | None = None) -> None:
        self.description = description
        self.task_type = task_type
        self.context = context or {}
