"""Execute AST nodes against an agent registry."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

from prompt_optimizer.grammar.ast_nodes import (
    ASTNode,
    AgentFieldRefNode,
    BlackboardRefNode,
    ConditionalNode,
    DirectiveNode,
    ParallelBlockNode,
    PipelineNode,
    PrevRefNode,
    ProgramNode,
    SequentialBlockNode,
)
from prompt_optimizer.runtime.context import ExecutionContext, StepResult
from prompt_optimizer.runtime.contracts import ContractEnforcer, ContractViolationError


class AgentAdapter(ABC):
    """Interface that bridges execution to your agent system.

    Implement this to connect execution to real agents.
    """

    @abstractmethod
    async def execute_directive(
        self,
        agent: str,
        action: str,
        target: Any,
        params: dict[str, Any],
        constraints: list[str],
        context: ExecutionContext,
    ) -> Any:
        """Execute a single directive and return the result."""
        ...

    @abstractmethod
    async def evaluate_condition(self, condition: str, context: ExecutionContext) -> bool:
        """Evaluate a conditional expression. Return True/False."""
        ...

    async def on_retry(self, agent: str, action: str, attempt: int, error: Exception) -> None:
        """Called before each retry attempt. Override for logging/metrics."""
        pass


class Executor:
    """Execute programs against an agent adapter."""

    def __init__(self, adapter: AgentAdapter) -> None:
        self._adapter = adapter
        self._contract_enforcer = ContractEnforcer()

    async def execute(self, node: ASTNode, context: ExecutionContext | None = None) -> Any:
        ctx = context or ExecutionContext()
        return await self._run(node, ctx)

    async def _run(self, node: ASTNode, ctx: ExecutionContext) -> Any:
        if isinstance(node, ProgramNode):
            return await self._run_program(node, ctx)
        if isinstance(node, DirectiveNode):
            return await self._run_directive(node, ctx)
        if isinstance(node, PipelineNode):
            return await self._run_pipeline(node, ctx)
        if isinstance(node, ParallelBlockNode):
            return await self._run_parallel(node, ctx)
        if isinstance(node, SequentialBlockNode):
            return await self._run_sequential(node, ctx)
        if isinstance(node, ConditionalNode):
            return await self._run_conditional(node, ctx)
        raise TypeError(f"Cannot execute node type: {type(node).__name__}")

    # ------------------------------------------------------------------
    # Program
    # ------------------------------------------------------------------

    async def _run_program(self, node: ProgramNode, ctx: ExecutionContext) -> Any:
        result = None
        for stmt in node.statements:
            result = await self._run(stmt, ctx)
        return result

    # ------------------------------------------------------------------
    # Directive
    # ------------------------------------------------------------------

    async def _run_directive(self, node: DirectiveNode, ctx: ExecutionContext) -> Any:
        agent = node.recipient.agent_code if node.recipient else ""
        target = self._resolve_value(node.target, ctx)

        params: dict[str, Any] = {}
        if node.params:
            for p in node.params.params:
                params[p.key] = p.value

        constraints: list[str] = []
        if node.constraints:
            constraints = [c.text for c in node.constraints.constraints]

        # Determine contract for validation
        contract = node.response_contract

        start = time.monotonic()
        last_error: Exception | None = None
        max_attempts = (node.retry.max_retries + 1) if node.retry else 1

        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    await self._adapter.on_retry(agent, node.action, attempt, last_error)
                    if node.retry:
                        delay = self._backoff_delay(node.retry.backoff, attempt)
                        await asyncio.sleep(delay)

                output = await self._adapter.execute_directive(
                    agent=agent,
                    action=node.action,
                    target=target,
                    params=params,
                    constraints=constraints,
                    context=ctx,
                )

                # Contract enforcement
                if contract and contract.fields:
                    violation = self._contract_enforcer.validate(output, contract)
                    if violation is not None:
                        raise ContractViolationError(violation)

                elapsed = (time.monotonic() - start) * 1000
                ctx.push_result(StepResult(
                    agent=agent,
                    action=node.action,
                    output=output,
                    success=True,
                    duration_ms=elapsed,
                ))
                return output

            except Exception as e:
                last_error = e

        # All retries exhausted — try fallback agent
        if node.retry and node.retry.fallback_agent:
            try:
                output = await self._adapter.execute_directive(
                    agent=node.retry.fallback_agent,
                    action=node.action,
                    target=target,
                    params=params,
                    constraints=constraints,
                    context=ctx,
                )
                elapsed = (time.monotonic() - start) * 1000
                ctx.push_result(StepResult(
                    agent=node.retry.fallback_agent,
                    action=node.action,
                    output=output,
                    success=True,
                    duration_ms=elapsed,
                ))
                return output
            except Exception as fallback_err:
                last_error = fallback_err

        # Final failure
        elapsed = (time.monotonic() - start) * 1000
        ctx.push_result(StepResult(
            agent=agent,
            action=node.action,
            success=False,
            error=str(last_error),
            duration_ms=elapsed,
        ))
        if max_attempts == 1 and last_error is not None:
            raise last_error
        raise RuntimeError(
            f"@{agent} {node.action} failed after {max_attempts} attempt(s): {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    async def _run_pipeline(self, node: PipelineNode, ctx: ExecutionContext) -> Any:
        result = None
        for step in node.directives:
            result = await self._run(step, ctx)
        return result

    async def _run_parallel(self, node: ParallelBlockNode, ctx: ExecutionContext) -> list[Any]:
        tasks = [self._run(branch, ctx) for branch in node.branches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Re-raise first error
        for r in results:
            if isinstance(r, Exception):
                raise r
        return list(results)

    async def _run_sequential(self, node: SequentialBlockNode, ctx: ExecutionContext) -> Any:
        result = None
        for step in node.steps:
            result = await self._run(step, ctx)
        return result

    async def _run_conditional(self, node: ConditionalNode, ctx: ExecutionContext) -> Any:
        # Get condition string
        if isinstance(node.condition, str):
            condition_str = node.condition
        else:
            # ConditionNode — reconstruct as string for adapter
            cond = node.condition
            if cond.comparator == ":" and cond.right:
                condition_str = f"{cond.left.field}:{cond.right.field}"
            elif cond.right:
                condition_str = f"{cond.left.field} {cond.comparator} {cond.right.field}"
            else:
                condition_str = cond.left.field

        condition_met = await self._adapter.evaluate_condition(condition_str, ctx)
        if condition_met:
            return await self._run(node.then_branch, ctx)
        elif node.else_branch:
            return await self._run(node.else_branch, ctx)
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_value(self, val: Any, ctx: ExecutionContext) -> Any:
        if isinstance(val, PrevRefNode):
            if val.step is not None:
                return ctx.get_prev(val.step)
            return ctx.prev
        if isinstance(val, BlackboardRefNode):
            return ctx.resolve_bb(val.namespace, val.key, val.version)
        if isinstance(val, AgentFieldRefNode):
            return f"@{val.agent}.{val.field}" if val.field else f"@{val.agent}"
        return val

    @staticmethod
    def _backoff_delay(strategy: str, attempt: int) -> float:
        if strategy == "linear":
            return attempt * 1.0
        if strategy == "fixed":
            return 1.0
        # exponential (default)
        return min(2 ** attempt * 0.1, 30.0)
