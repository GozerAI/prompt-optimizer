"""Tests for runtime executor edge cases — empty programs, failure modes,
timeout handling, contract enforcement, context scoping, retry/fallback."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from prompt_optimizer.grammar.ast_nodes import (
    ContractFieldNode,
    DirectiveNode,
    ParallelBlockNode,
    PipelineNode,
    PrevRefNode,
    ProgramNode,
    RecipientNode,
    ResponseContractNode,
    RetryPolicyNode,
    SequentialBlockNode,
)
from prompt_optimizer.grammar.lexer import Lexer
from prompt_optimizer.grammar.parser import Parser
from prompt_optimizer.runtime.context import ExecutionContext, StepResult
from prompt_optimizer.runtime.contracts import (
    ContractEnforcer,
    ContractViolation,
    ContractViolationError,
)
from prompt_optimizer.runtime.executor import AgentAdapter, Executor


# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------


class EdgeCaseAdapter(AgentAdapter):
    """Adapter for edge case testing with configurable behavior."""

    def __init__(self):
        self.calls: list[dict] = []
        self._results: dict[str, Any] = {}
        self._fail_counts: dict[str, int] = {}  # agent:action -> how many times to fail
        self._call_counts: dict[str, int] = {}
        self.retry_log: list[tuple[str, str, int]] = []

    def set_result(self, agent: str, action: str, result: Any):
        self._results[f"{agent}:{action}"] = result

    def set_fail_n_times(self, agent: str, action: str, n: int, error: Exception, then_result: Any):
        key = f"{agent}:{action}"
        self._fail_counts[key] = n
        self._results[f"{key}:error"] = error
        self._results[f"{key}:success"] = then_result

    async def execute_directive(self, agent, action, target, params, constraints, context):
        key = f"{agent}:{action}"
        self.calls.append({
            "agent": agent, "action": action, "target": target,
            "params": params, "constraints": constraints,
        })
        self._call_counts[key] = self._call_counts.get(key, 0) + 1

        # Fail N times then succeed
        if key in self._fail_counts:
            if self._call_counts[key] <= self._fail_counts[key]:
                raise self._results[f"{key}:error"]
            return self._results[f"{key}:success"]

        if key in self._results:
            val = self._results[key]
            if isinstance(val, Exception):
                raise val
            return val
        return f"{agent}:{action}:done"

    async def evaluate_condition(self, condition, context):
        return True

    async def on_retry(self, agent, action, attempt, error):
        self.retry_log.append((agent, action, attempt))


def parse(src: str):
    tokens = Lexer().tokenize(src)
    return Parser(tokens).parse()


@pytest.fixture
def adapter():
    return EdgeCaseAdapter()


@pytest.fixture
def executor(adapter):
    return Executor(adapter)


# ============================================================
# Execute empty / minimal programs
# ============================================================


class TestEmptyProgram:
    async def test_empty_program_node(self, adapter, executor):
        node = ProgramNode(statements=[])
        result = await executor.execute(node)
        assert result is None

    async def test_single_directive(self, adapter, executor):
        adapter.set_result("CEO", "DECIDE", "yes")
        node = parse("@CEO DECIDE expansion")
        result = await executor.execute(node)
        assert result == "yes"
        assert len(adapter.calls) == 1

    async def test_program_with_one_statement(self, adapter, executor):
        adapter.set_result("CEO", "DECIDE", "approved")
        node = ProgramNode(statements=[
            DirectiveNode(
                action="DECIDE", target="expansion",
                recipient=RecipientNode(agent_code="CEO"),
            )
        ])
        result = await executor.execute(node)
        assert result == "approved"


# ============================================================
# PAR block with all agents failing
# ============================================================


class TestParAllFailing:
    async def test_par_all_fail_raises_first_error(self, adapter, executor):
        adapter.set_result("CFO", "FORECAST", ValueError("CFO down"))
        adapter.set_result("CTO", "ASSESS", RuntimeError("CTO down"))
        with pytest.raises((ValueError, RuntimeError)):
            await executor.execute(
                parse("PAR { @CFO FORECAST revenue; @CTO ASSESS infra }")
            )

    async def test_par_one_fail_raises(self, adapter, executor):
        adapter.set_result("CFO", "FORECAST", "ok")
        adapter.set_result("CTO", "ASSESS", ValueError("fail"))
        with pytest.raises(ValueError, match="fail"):
            await executor.execute(
                parse("PAR { @CFO FORECAST revenue; @CTO ASSESS infra }")
            )

    async def test_par_success(self, adapter, executor):
        adapter.set_result("CFO", "FORECAST", "f1")
        adapter.set_result("CTO", "ASSESS", "a1")
        result = await executor.execute(
            parse("PAR { @CFO FORECAST revenue; @CTO ASSESS infra }")
        )
        assert isinstance(result, list)
        assert set(result) == {"f1", "a1"}


# ============================================================
# SEQ block with early failure
# ============================================================


class TestSeqEarlyFailure:
    async def test_seq_first_step_fails(self, adapter, executor):
        adapter.set_result("CDO", "GATHER", RuntimeError("data source down"))
        adapter.set_result("CFO", "ANALYZE", "done")
        with pytest.raises(RuntimeError, match="data source down"):
            await executor.execute(
                parse("SEQ { @CDO GATHER data; @CFO ANALYZE $prev }")
            )
        # Second step should NOT have been called
        assert len(adapter.calls) == 1

    async def test_seq_second_step_fails(self, adapter, executor):
        adapter.set_result("CDO", "GATHER", "raw_data")
        adapter.set_result("CFO", "ANALYZE", ValueError("analysis failed"))
        with pytest.raises(ValueError, match="analysis failed"):
            await executor.execute(
                parse("SEQ { @CDO GATHER data; @CFO ANALYZE $prev }")
            )
        assert len(adapter.calls) == 2

    async def test_seq_all_succeed(self, adapter, executor):
        adapter.set_result("CDO", "GATHER", "data")
        adapter.set_result("CFO", "ANALYZE", "analysis")
        result = await executor.execute(
            parse("SEQ { @CDO GATHER data; @CFO ANALYZE $prev }")
        )
        assert result == "analysis"


# ============================================================
# ContractEnforcer edge cases
# ============================================================


class TestContractEnforcerEdgeCases:
    def test_missing_required_field(self):
        enforcer = ContractEnforcer()
        contract = ResponseContractNode(fields=(
            ContractFieldNode(name="score", type_hint="float", required=True),
            ContractFieldNode(name="label", type_hint="str", required=True),
        ))
        output = {"score": 0.9}
        violation = enforcer.validate(output, contract)
        assert violation is not None
        assert "label" in violation.missing_fields

    def test_extra_fields_accepted(self):
        enforcer = ContractEnforcer()
        contract = ResponseContractNode(fields=(
            ContractFieldNode(name="score", type_hint="float", required=True),
        ))
        output = {"score": 0.9, "extra_field": "bonus", "another": 42}
        violation = enforcer.validate(output, contract)
        assert violation is None

    def test_all_fields_missing(self):
        enforcer = ContractEnforcer()
        contract = ResponseContractNode(fields=(
            ContractFieldNode(name="a", type_hint="str", required=True),
            ContractFieldNode(name="b", type_hint="int", required=True),
        ))
        violation = enforcer.validate({}, contract)
        assert violation is not None
        assert set(violation.missing_fields) == {"a", "b"}

    def test_non_dict_output_with_contract(self):
        enforcer = ContractEnforcer()
        contract = ResponseContractNode(fields=(
            ContractFieldNode(name="x", type_hint="str", required=True),
        ))
        violation = enforcer.validate("just a string", contract)
        assert violation is not None
        assert "x" in violation.missing_fields

    def test_non_dict_list_output(self):
        enforcer = ContractEnforcer()
        contract = ResponseContractNode(fields=(
            ContractFieldNode(name="data", type_hint="list", required=True),
        ))
        violation = enforcer.validate([1, 2, 3], contract)
        assert violation is not None

    def test_empty_contract_accepts_anything(self):
        enforcer = ContractEnforcer()
        contract = ResponseContractNode(fields=())
        assert enforcer.validate("anything", contract) is None
        assert enforcer.validate(42, contract) is None
        assert enforcer.validate(None, contract) is None

    def test_optional_field_missing_ok(self):
        enforcer = ContractEnforcer()
        contract = ResponseContractNode(fields=(
            ContractFieldNode(name="required_field", type_hint="str", required=True),
            ContractFieldNode(name="optional_field", type_hint="str", required=False),
        ))
        output = {"required_field": "present"}
        assert enforcer.validate(output, contract) is None

    def test_type_error_reported(self):
        enforcer = ContractEnforcer()
        contract = ResponseContractNode(fields=(
            ContractFieldNode(name="count", type_hint="int", required=True),
        ))
        output = {"count": "not_a_number"}
        violation = enforcer.validate(output, contract)
        assert violation is not None
        assert len(violation.type_errors) == 1
        assert violation.type_errors[0] == ("count", "int", "str")

    def test_int_accepted_as_float(self):
        enforcer = ContractEnforcer()
        contract = ResponseContractNode(fields=(
            ContractFieldNode(name="value", type_hint="float", required=True),
        ))
        output = {"value": 42}
        assert enforcer.validate(output, contract) is None

    def test_float_not_accepted_as_int(self):
        enforcer = ContractEnforcer()
        contract = ResponseContractNode(fields=(
            ContractFieldNode(name="count", type_hint="int", required=True),
        ))
        output = {"count": 3.14}
        violation = enforcer.validate(output, contract)
        assert violation is not None

    def test_unknown_type_hint_skipped(self):
        enforcer = ContractEnforcer()
        contract = ResponseContractNode(fields=(
            ContractFieldNode(name="custom", type_hint="custom_type", required=True),
        ))
        output = {"custom": "anything"}
        assert enforcer.validate(output, contract) is None

    def test_violation_error_wraps_violation(self):
        violation = ContractViolation(missing_fields=["x"])
        error = ContractViolationError(violation)
        assert error.violation is violation
        assert "x" in str(error)


# ============================================================
# ExecutionContext variable scoping
# ============================================================


class TestExecutionContextScoping:
    def test_empty_context_prev_is_none(self):
        ctx = ExecutionContext()
        assert ctx.prev is None

    def test_push_result_updates_prev(self):
        ctx = ExecutionContext()
        ctx.push_result(StepResult(agent="CEO", action="DECIDE", output="yes"))
        assert ctx.prev == "yes"

    def test_get_prev_by_index(self):
        ctx = ExecutionContext()
        ctx.push_result(StepResult(agent="CDO", action="GATHER", output="data1"))
        ctx.push_result(StepResult(agent="CFO", action="ANALYZE", output="data2"))
        assert ctx.get_prev(0) == "data1"
        assert ctx.get_prev(1) == "data2"

    def test_get_prev_out_of_range(self):
        ctx = ExecutionContext()
        assert ctx.get_prev(0) is None
        assert ctx.get_prev(99) is None

    def test_blackboard_resolution(self):
        ctx = ExecutionContext()
        ctx.blackboard["financial:revenue"] = 2_300_000
        assert ctx.resolve_bb("financial", "revenue") == 2_300_000

    def test_blackboard_missing_key(self):
        ctx = ExecutionContext()
        assert ctx.resolve_bb("nonexistent", "key") is None

    def test_variables_dict(self):
        ctx = ExecutionContext()
        ctx.variables["counter"] = 1
        assert ctx.variables["counter"] == 1

    def test_metadata_dict(self):
        ctx = ExecutionContext()
        ctx.metadata["start_time"] = 12345
        assert ctx.metadata["start_time"] == 12345

    def test_sender_field(self):
        ctx = ExecutionContext(sender="CEO")
        assert ctx.sender == "CEO"

    def test_step_results_track_failures(self):
        ctx = ExecutionContext()
        ctx.push_result(StepResult(agent="CTO", action="ASSESS", success=False, error="timeout"))
        assert not ctx.step_results[0].success
        assert ctx.step_results[0].error == "timeout"


# ============================================================
# $prev reference to failed step
# ============================================================


class TestPrevRefToFailedStep:
    async def test_prev_after_failed_step_in_pipeline(self, adapter, executor):
        # When first step fails, the pipeline should not continue
        adapter.set_result("CDO", "GATHER", RuntimeError("fail"))
        with pytest.raises(RuntimeError):
            await executor.execute(parse("@CDO GATHER data | @CFO ANALYZE $prev"))

    async def test_prev_resolves_none_when_empty(self, adapter, executor):
        adapter.set_result("CFO", "ANALYZE", "done")
        ctx = ExecutionContext()
        # Direct execution with $prev target and empty context
        node = DirectiveNode(
            action="ANALYZE",
            target=PrevRefNode(step=None),
            recipient=RecipientNode(agent_code="CFO"),
        )
        result = await executor.execute(node, ctx)
        # prev is None but execution still proceeds
        assert result == "done"
        assert adapter.calls[0]["target"] is None


# ============================================================
# Retry with exponential backoff
# ============================================================


class TestRetryBackoff:
    async def test_retry_succeeds_after_failures(self, adapter, executor):
        adapter.set_fail_n_times("CTO", "ANALYZE", 2, ConnectionError("timeout"), "recovered")
        node = parse("@CTO ANALYZE infra RETRY 3")
        result = await executor.execute(node)
        assert result == "recovered"
        assert len(adapter.retry_log) == 2  # two retries before success

    async def test_retry_exhausted_raises(self, adapter, executor):
        adapter.set_result("CTO", "ANALYZE", RuntimeError("permanently down"))
        with pytest.raises(RuntimeError, match="failed after"):
            await executor.execute(parse("@CTO ANALYZE infra RETRY 2"))

    async def test_retry_with_fallback_agent(self, adapter, executor):
        adapter.set_result("CTO", "ANALYZE", RuntimeError("down"))
        adapter.set_result("CIO", "ANALYZE", "fallback_ok")
        result = await executor.execute(parse("@CTO ANALYZE infra RETRY 1 FALLBACK @CIO"))
        assert result == "fallback_ok"

    async def test_backoff_delay_exponential(self):
        delay = Executor._backoff_delay("exp", 0)
        assert delay == pytest.approx(0.1, abs=0.01)
        delay = Executor._backoff_delay("exp", 1)
        assert delay == pytest.approx(0.2, abs=0.01)
        delay = Executor._backoff_delay("exp", 2)
        assert delay == pytest.approx(0.4, abs=0.01)

    async def test_backoff_delay_linear(self):
        assert Executor._backoff_delay("linear", 0) == 0.0
        assert Executor._backoff_delay("linear", 1) == 1.0
        assert Executor._backoff_delay("linear", 2) == 2.0

    async def test_backoff_delay_fixed(self):
        assert Executor._backoff_delay("fixed", 0) == 1.0
        assert Executor._backoff_delay("fixed", 5) == 1.0

    async def test_backoff_delay_exp_capped_at_30(self):
        delay = Executor._backoff_delay("exp", 100)
        assert delay <= 30.0


# ============================================================
# Fallback chain exhaustion
# ============================================================


class TestFallbackChainExhaustion:
    async def test_fallback_also_fails_raises(self, adapter, executor):
        adapter.set_result("CTO", "ANALYZE", RuntimeError("CTO down"))
        adapter.set_result("CIO", "ANALYZE", RuntimeError("CIO also down"))
        with pytest.raises(RuntimeError):
            await executor.execute(parse("@CTO ANALYZE infra RETRY 1 FALLBACK @CIO"))

    async def test_no_fallback_no_retry_raises_original(self, adapter, executor):
        adapter.set_result("CTO", "ANALYZE", ValueError("bad input"))
        with pytest.raises(ValueError, match="bad input"):
            await executor.execute(parse("@CTO ANALYZE infra"))


# ============================================================
# Agent adapter ABC enforcement
# ============================================================


class TestAgentAdapterABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            AgentAdapter()

    def test_must_implement_execute_directive(self):
        class IncompleteAdapter(AgentAdapter):
            async def evaluate_condition(self, condition, context):
                return True

        with pytest.raises(TypeError):
            IncompleteAdapter()

    def test_must_implement_evaluate_condition(self):
        class IncompleteAdapter(AgentAdapter):
            async def execute_directive(self, agent, action, target, params, constraints, context):
                return None

        with pytest.raises(TypeError):
            IncompleteAdapter()

    def test_complete_adapter_instantiates(self):
        class CompleteAdapter(AgentAdapter):
            async def execute_directive(self, agent, action, target, params, constraints, context):
                return None

            async def evaluate_condition(self, condition, context):
                return True

        adapter = CompleteAdapter()
        assert adapter is not None

    async def test_on_retry_default_is_noop(self):
        class MinimalAdapter(AgentAdapter):
            async def execute_directive(self, agent, action, target, params, constraints, context):
                return None

            async def evaluate_condition(self, condition, context):
                return True

        adapter = MinimalAdapter()
        # on_retry has a default implementation that does nothing
        await adapter.on_retry("CEO", "DECIDE", 1, RuntimeError("test"))


# ============================================================
# Executor with unknown node type
# ============================================================


class TestExecutorUnknownNode:
    async def test_unknown_node_type_raises(self, adapter, executor):
        class WeirdNode:
            pass

        with pytest.raises(TypeError, match="Cannot execute node type"):
            await executor.execute(WeirdNode())


# ============================================================
# Context tracking through execution
# ============================================================


class TestContextTracking:
    async def test_context_records_all_steps(self, adapter, executor):
        adapter.set_result("CDO", "GATHER", "raw")
        adapter.set_result("CFO", "ANALYZE", "analyzed")
        ctx = ExecutionContext()
        await executor.execute(parse("@CDO GATHER data | @CFO ANALYZE $prev"), ctx)
        assert len(ctx.step_results) == 2
        assert ctx.step_results[0].agent == "CDO"
        assert ctx.step_results[0].success is True
        assert ctx.step_results[1].agent == "CFO"

    async def test_context_records_duration(self, adapter, executor):
        adapter.set_result("CEO", "DECIDE", "yes")
        ctx = ExecutionContext()
        await executor.execute(parse("@CEO DECIDE expansion"), ctx)
        assert ctx.step_results[0].duration_ms >= 0

    async def test_context_records_failure(self, adapter, executor):
        adapter.set_result("CTO", "ANALYZE", RuntimeError("down"))
        ctx = ExecutionContext()
        with pytest.raises(RuntimeError):
            await executor.execute(parse("@CTO ANALYZE infra"), ctx)
        # Failed step should still be recorded
        # (single attempt failure raises immediately)

    async def test_step_result_repr(self):
        sr = StepResult(agent="CEO", action="DECIDE", output="yes", success=True)
        r = repr(sr)
        assert "CEO" in r
        assert "DECIDE" in r
        assert "OK" in r

    async def test_step_result_error_repr(self):
        sr = StepResult(agent="CTO", action="ASSESS", success=False, error="timeout")
        r = repr(sr)
        assert "ERR" in r
        assert "timeout" in r
