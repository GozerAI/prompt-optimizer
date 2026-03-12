"""Tests for the prompt-optimizer executor (ported from AIL)."""

import asyncio
import pytest
from typing import Any

from prompt_optimizer.grammar.lexer import Lexer
from prompt_optimizer.grammar.parser import Parser
from prompt_optimizer.runtime.executor import AgentAdapter, Executor
from prompt_optimizer.runtime.context import ExecutionContext


class MockAdapter(AgentAdapter):
    """Mock adapter that records calls and returns canned results."""

    def __init__(self, results: dict[str, Any] | None = None):
        self.calls: list[dict] = []
        self._results = results or {}
        self._condition_results: dict[str, bool] = {}
        self.retry_log: list[tuple[str, str, int]] = []

    def set_result(self, agent: str, action: str, result: Any):
        self._results[f"{agent}:{action}"] = result

    def set_condition(self, condition_fragment: str, result: bool):
        self._condition_results[condition_fragment] = result

    async def execute_directive(self, agent, action, target, params, constraints, context):
        self.calls.append({
            "agent": agent, "action": action, "target": target,
            "params": params, "constraints": constraints,
        })
        key = f"{agent}:{action}"
        if key in self._results:
            val = self._results[key]
            if isinstance(val, Exception):
                raise val
            return val
        return f"{agent}:{action}:done"

    async def evaluate_condition(self, condition, context):
        for frag, result in self._condition_results.items():
            if frag in condition:
                return result
        return True

    async def on_retry(self, agent, action, attempt, error):
        self.retry_log.append((agent, action, attempt))


def parse(src: str):
    tokens = Lexer().tokenize(src)
    return Parser(tokens).parse()


@pytest.fixture
def adapter():
    return MockAdapter()


@pytest.fixture
def executor(adapter):
    return Executor(adapter)


class TestDirectiveExecution:
    @pytest.mark.asyncio
    async def test_simple(self, adapter, executor):
        adapter.set_result("CEO", "DECIDE", "approved")
        result = await executor.execute(parse("@CEO DECIDE expansion"))
        assert result == "approved"
        assert len(adapter.calls) == 1
        assert adapter.calls[0]["agent"] == "CEO"

    @pytest.mark.asyncio
    async def test_with_params(self, adapter, executor):
        adapter.set_result("CFO", "ANALYZE", {"revenue": 2.3})
        result = await executor.execute(parse("@CFO ANALYZE revenue {period=Q1}"))
        assert result == {"revenue": 2.3}
        assert adapter.calls[0]["params"] == {"period": "Q1"}

    @pytest.mark.asyncio
    async def test_with_constraints(self, adapter, executor):
        adapter.set_result("CFO", "ANALYZE", "ok")
        await executor.execute(parse("@CFO ANALYZE revenue [margin > 20%]"))
        assert "margin > 20%" in adapter.calls[0]["constraints"]


class TestPipelineExecution:
    @pytest.mark.asyncio
    async def test_two_step(self, adapter, executor):
        adapter.set_result("CDO", "GATHER", {"data": [1, 2, 3]})
        adapter.set_result("CFO", "ANALYZE", "analysis_complete")
        result = await executor.execute(parse("@CDO GATHER data | @CFO ANALYZE $prev"))
        assert result == "analysis_complete"
        assert len(adapter.calls) == 2
        # Second step should receive prev result as target
        assert adapter.calls[1]["target"] == {"data": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_three_step(self, adapter, executor):
        adapter.set_result("CDO", "GATHER", "raw_data")
        adapter.set_result("CFO", "ANALYZE", "analysis")
        adapter.set_result("CEO", "DECIDE", "go")
        result = await executor.execute(
            parse("@CDO GATHER metrics | @CFO ANALYZE $prev | @CEO DECIDE $prev")
        )
        assert result == "go"
        assert adapter.calls[2]["target"] == "analysis"


class TestParallelExecution:
    @pytest.mark.asyncio
    async def test_parallel(self, adapter, executor):
        adapter.set_result("CFO", "FORECAST", "forecast_done")
        adapter.set_result("CTO", "ASSESS", "assess_done")
        result = await executor.execute(
            parse("PAR { @CFO FORECAST revenue; @CTO ASSESS infra }")
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert set(result) == {"forecast_done", "assess_done"}

    @pytest.mark.asyncio
    async def test_parallel_error_propagates(self, adapter, executor):
        adapter.set_result("CFO", "FORECAST", "ok")
        adapter.set_result("CTO", "ASSESS", ValueError("infra down"))
        with pytest.raises(ValueError, match="infra down"):
            await executor.execute(
                parse("PAR { @CFO FORECAST revenue; @CTO ASSESS infra }")
            )


class TestSequentialExecution:
    @pytest.mark.asyncio
    async def test_sequential(self, adapter, executor):
        adapter.set_result("CDO", "GATHER", "data")
        adapter.set_result("CFO", "ANALYZE", "result")
        result = await executor.execute(
            parse("SEQ { @CDO GATHER data; @CFO ANALYZE $prev }")
        )
        assert result == "result"
        assert adapter.calls[1]["target"] == "data"


class TestConditionalExecution:
    @pytest.mark.asyncio
    async def test_condition_true(self, adapter, executor):
        adapter.set_condition("pipeline", True)
        adapter.set_result("CEO", "DECIDE", "expand")
        result = await executor.execute(
            parse("IF @CRO.pipeline > 1M THEN @CEO DECIDE expansion")
        )
        assert result == "expand"

    @pytest.mark.asyncio
    async def test_condition_false_with_else(self, adapter, executor):
        adapter.set_condition("pipeline", False)
        adapter.set_result("CFO", "ANALYZE", "cost_analysis")
        result = await executor.execute(
            parse("IF @CRO.pipeline > 1M THEN @CEO DECIDE expand ELSE @CFO ANALYZE cuts")
        )
        assert result == "cost_analysis"

    @pytest.mark.asyncio
    async def test_condition_false_no_else(self, adapter, executor):
        adapter.set_condition("pipeline", False)
        result = await executor.execute(
            parse("IF @CRO.pipeline > 1M THEN @CEO DECIDE expand")
        )
        assert result is None


class TestRetryExecution:
    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self, adapter, executor):
        call_count = 0

        async def flaky(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("timeout")
            return "recovered"

        adapter.execute_directive = flaky
        result = await executor.execute(parse("@CTO ANALYZE infra RETRY 3"))
        assert result == "recovered"

    @pytest.mark.asyncio
    async def test_retry_exhausted_with_fallback(self, adapter, executor):
        adapter.set_result("CTO", "ANALYZE", RuntimeError("down"))
        adapter.set_result("CIO", "ANALYZE", "fallback_result")
        result = await executor.execute(parse("@CTO ANALYZE infra RETRY 1 FALLBACK @CIO"))
        assert result == "fallback_result"

    @pytest.mark.asyncio
    async def test_retry_exhausted_no_fallback_raises(self, adapter, executor):
        adapter.set_result("CTO", "ANALYZE", RuntimeError("down"))
        with pytest.raises(RuntimeError, match="failed after"):
            await executor.execute(parse("@CTO ANALYZE infra RETRY 1"))


class TestContext:
    @pytest.mark.asyncio
    async def test_context_tracks_results(self, adapter, executor):
        adapter.set_result("CDO", "GATHER", "raw")
        adapter.set_result("CFO", "ANALYZE", "done")
        ctx = ExecutionContext()
        await executor.execute(
            parse("@CDO GATHER data | @CFO ANALYZE $prev"), ctx
        )
        assert len(ctx.step_results) == 2
        assert ctx.step_results[0].output == "raw"
        assert ctx.step_results[1].output == "done"
        assert all(r.success for r in ctx.step_results)

    @pytest.mark.asyncio
    async def test_blackboard_resolution(self, adapter, executor):
        adapter.set_result("CFO", "ANALYZE", "analysis")
        ctx = ExecutionContext()
        ctx.blackboard["financial:revenue"] = 2_300_000
        result = await executor.execute(
            parse("@CFO ANALYZE bb:financial:revenue"), ctx
        )
        assert result == "analysis"
        assert adapter.calls[0]["target"] == 2_300_000


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_complex_workflow(self, adapter, executor):
        """Full workflow: gather in parallel, analyze, decide."""
        adapter.set_result("CFO", "FORECAST", {"revenue": "10M"})
        adapter.set_result("CTO", "ASSESS", {"infra": "healthy"})
        adapter.set_result("CDO", "GATHER", "market_data")

        par_result = await executor.execute(
            parse("PAR { @CFO FORECAST revenue; @CTO ASSESS infra; @CDO GATHER market }")
        )
        assert len(par_result) == 3
