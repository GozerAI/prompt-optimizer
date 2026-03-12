"""Tests for response contract enforcement (ported from AIL)."""

import pytest
from typing import Any

from prompt_optimizer.grammar.ast_nodes import ContractFieldNode, ResponseContractNode
from prompt_optimizer.runtime.contracts import ContractEnforcer, ContractViolation, ContractViolationError
from prompt_optimizer.runtime.executor import AgentAdapter, Executor
from prompt_optimizer.runtime.context import ExecutionContext
from prompt_optimizer.grammar.lexer import Lexer
from prompt_optimizer.grammar.parser import Parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_contract(*fields: tuple[str, str] | tuple[str, str, bool]) -> ResponseContractNode:
    """Shorthand to build a ResponseContractNode."""
    cfs = []
    for f in fields:
        if len(f) == 2:
            cfs.append(ContractFieldNode(name=f[0], type_hint=f[1], required=True))
        else:
            cfs.append(ContractFieldNode(name=f[0], type_hint=f[1], required=f[2]))
    return ResponseContractNode(fields=tuple(cfs))


class ContractMockAdapter(AgentAdapter):
    """Adapter that returns a configurable output dict."""

    def __init__(self) -> None:
        self.output: Any = {}
        self.call_count: int = 0

    async def execute_directive(self, agent, action, target, params, constraints, context):
        self.call_count += 1
        if isinstance(self.output, Exception):
            raise self.output
        return self.output

    async def evaluate_condition(self, condition, context):
        return True


def parse(src: str):
    tokens = Lexer().tokenize(src)
    return Parser(tokens).parse()


# ---------------------------------------------------------------------------
# Unit tests: ContractEnforcer.validate
# ---------------------------------------------------------------------------

class TestContractEnforcerValid:
    """Tests for outputs that satisfy the contract."""

    def test_all_required_fields_present(self):
        enforcer = ContractEnforcer()
        contract = make_contract(("score", "float"), ("label", "str"))
        output = {"score": 0.95, "label": "positive"}
        assert enforcer.validate(output, contract) is None

    def test_int_accepted_as_float(self):
        enforcer = ContractEnforcer()
        contract = make_contract(("value", "float"))
        output = {"value": 42}
        assert enforcer.validate(output, contract) is None

    def test_any_type_accepts_anything(self):
        enforcer = ContractEnforcer()
        contract = make_contract(("data", "any"))
        output = {"data": [1, 2, 3]}
        assert enforcer.validate(output, contract) is None

    def test_optional_field_present(self):
        enforcer = ContractEnforcer()
        contract = make_contract(("name", "str"), ("notes", "str", False))
        output = {"name": "test", "notes": "extra"}
        assert enforcer.validate(output, contract) is None

    def test_optional_field_missing(self):
        enforcer = ContractEnforcer()
        contract = make_contract(("name", "str"), ("notes", "str", False))
        output = {"name": "test"}
        assert enforcer.validate(output, contract) is None

    def test_empty_contract_always_valid(self):
        enforcer = ContractEnforcer()
        contract = ResponseContractNode()
        output = "anything"
        assert enforcer.validate(output, contract) is None

    def test_dict_type_field(self):
        enforcer = ContractEnforcer()
        contract = make_contract(("metadata", "dict"))
        output = {"metadata": {"key": "val"}}
        assert enforcer.validate(output, contract) is None

    def test_list_type_field(self):
        enforcer = ContractEnforcer()
        contract = make_contract(("items", "list"))
        output = {"items": [1, 2, 3]}
        assert enforcer.validate(output, contract) is None

    def test_bool_type_field(self):
        enforcer = ContractEnforcer()
        contract = make_contract(("active", "bool"))
        output = {"active": True}
        assert enforcer.validate(output, contract) is None

    def test_int_type_field(self):
        enforcer = ContractEnforcer()
        contract = make_contract(("count", "int"))
        output = {"count": 5}
        assert enforcer.validate(output, contract) is None


class TestContractEnforcerInvalid:
    """Tests for outputs that violate the contract."""

    def test_missing_required_field(self):
        enforcer = ContractEnforcer()
        contract = make_contract(("score", "float"), ("label", "str"))
        output = {"score": 0.95}
        violation = enforcer.validate(output, contract)
        assert violation is not None
        assert "label" in violation.missing_fields

    def test_multiple_missing_fields(self):
        enforcer = ContractEnforcer()
        contract = make_contract(("a", "str"), ("b", "int"), ("c", "float"))
        output = {}
        violation = enforcer.validate(output, contract)
        assert violation is not None
        assert set(violation.missing_fields) == {"a", "b", "c"}

    def test_type_mismatch_str_got_int(self):
        enforcer = ContractEnforcer()
        contract = make_contract(("name", "str"))
        output = {"name": 42}
        violation = enforcer.validate(output, contract)
        assert violation is not None
        assert len(violation.type_errors) == 1
        assert violation.type_errors[0] == ("name", "str", "int")

    def test_type_mismatch_int_got_str(self):
        enforcer = ContractEnforcer()
        contract = make_contract(("count", "int"))
        output = {"count": "five"}
        violation = enforcer.validate(output, contract)
        assert violation is not None
        assert ("count", "int", "str") in violation.type_errors

    def test_type_mismatch_list_got_dict(self):
        enforcer = ContractEnforcer()
        contract = make_contract(("items", "list"))
        output = {"items": {"a": 1}}
        violation = enforcer.validate(output, contract)
        assert violation is not None
        assert ("items", "list", "dict") in violation.type_errors

    def test_non_dict_output(self):
        enforcer = ContractEnforcer()
        contract = make_contract(("field", "str"))
        output = "just a string"
        violation = enforcer.validate(output, contract)
        assert violation is not None
        assert "field" in violation.missing_fields

    def test_non_dict_output_message(self):
        enforcer = ContractEnforcer()
        contract = make_contract(("field", "str"))
        violation = enforcer.validate(42, contract)
        assert violation is not None
        assert "dict" in violation.message.lower() or "int" in violation.message.lower()

    def test_mixed_missing_and_type_errors(self):
        enforcer = ContractEnforcer()
        contract = make_contract(("name", "str"), ("score", "float"))
        output = {"name": 123}  # name wrong type, score missing
        violation = enforcer.validate(output, contract)
        assert violation is not None
        assert "score" in violation.missing_fields
        assert ("name", "str", "int") in violation.type_errors

    def test_violation_message_contains_details(self):
        enforcer = ContractEnforcer()
        contract = make_contract(("x", "str"))
        output = {}
        violation = enforcer.validate(output, contract)
        assert violation is not None
        assert "x" in violation.message

    def test_unknown_type_hint_skipped(self):
        enforcer = ContractEnforcer()
        contract = make_contract(("data", "custom_type"))
        output = {"data": "anything"}
        # Unknown type hints are not validated
        assert enforcer.validate(output, contract) is None


# ---------------------------------------------------------------------------
# Integration tests: executor + contracts
# ---------------------------------------------------------------------------

class TestContractExecutorIntegration:

    @pytest.mark.asyncio
    async def test_valid_output_passes(self):
        adapter = ContractMockAdapter()
        adapter.output = {"confidence": 0.9, "recommendation": "buy"}
        executor = Executor(adapter)
        node = parse("@CFO ANALYZE revenue -> {confidence: float, recommendation: str}")
        result = await executor.execute(node)
        assert result == {"confidence": 0.9, "recommendation": "buy"}

    @pytest.mark.asyncio
    async def test_missing_field_raises_violation(self):
        adapter = ContractMockAdapter()
        adapter.output = {"confidence": 0.9}  # missing recommendation
        executor = Executor(adapter)
        node = parse("@CFO ANALYZE revenue -> {confidence: float, recommendation: str}")
        with pytest.raises(ContractViolationError) as exc_info:
            await executor.execute(node)
        assert "recommendation" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_type_error_raises_violation(self):
        adapter = ContractMockAdapter()
        adapter.output = {"confidence": "high", "recommendation": "buy"}
        executor = Executor(adapter)
        node = parse("@CFO ANALYZE revenue -> {confidence: float, recommendation: str}")
        with pytest.raises(ContractViolationError):
            await executor.execute(node)

    @pytest.mark.asyncio
    async def test_optional_field_missing_ok(self):
        adapter = ContractMockAdapter()
        adapter.output = {"score": 0.8}
        executor = Executor(adapter)
        node = parse("@CFO ANALYZE revenue -> {score: float, notes?: str}")
        result = await executor.execute(node)
        assert result == {"score": 0.8}

    @pytest.mark.asyncio
    async def test_retry_applies_to_contract_violations(self):
        """Contract violations should be retried like any other error."""
        call_count = 0
        adapter = ContractMockAdapter()

        async def improving_output(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"wrong": "output"}  # will violate contract
            return {"score": 0.95}  # correct on retry

        adapter.execute_directive = improving_output
        executor = Executor(adapter)
        node = parse("@CFO ANALYZE revenue -> {score: float} RETRY 3")
        result = await executor.execute(node)
        assert result == {"score": 0.95}
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_contract_no_validation(self):
        """Directives without contracts should not be validated."""
        adapter = ContractMockAdapter()
        adapter.output = "plain string"
        executor = Executor(adapter)
        node = parse("@CEO DECIDE expansion")
        result = await executor.execute(node)
        assert result == "plain string"

    @pytest.mark.asyncio
    async def test_format_hint_only_no_validation(self):
        """Response with only format_hint (no fields) should skip validation."""
        adapter = ContractMockAdapter()
        adapter.output = "summary text"
        executor = Executor(adapter)
        node = parse("@CEO DECIDE expansion -> summary")
        result = await executor.execute(node)
        assert result == "summary text"
