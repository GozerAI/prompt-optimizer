"""Extended parser tests for orchestration features (ported from AIL)."""

import pytest
from prompt_optimizer.grammar.lexer import Lexer
from prompt_optimizer.grammar.parser import Parser
from prompt_optimizer.grammar.ast_nodes import (
    BlackboardRefNode,
    ConditionalNode,
    ContractFieldNode,
    DirectiveNode,
    ParallelBlockNode,
    PipelineNode,
    PrevRefNode,
    ProgramNode,
    ResponseContractNode,
    RetryPolicyNode,
    SequentialBlockNode,
    AgentFieldRefNode,
)


def parse(src: str):
    tokens = Lexer().tokenize(src)
    return Parser(tokens).parse()


class TestSimpleDirective:
    def test_minimal(self):
        node = parse("@CEO DECIDE")
        assert isinstance(node, DirectiveNode)
        assert node.recipient.agent_code == "CEO"
        assert node.action == "DECIDE"

    def test_with_target(self):
        node = parse("@CFO ANALYZE revenue")
        assert isinstance(node, DirectiveNode)
        assert node.target == "revenue"

    def test_with_params(self):
        node = parse("@CFO ANALYZE revenue {period=Q1, depth=detailed}")
        params = {p.key: p.value for p in node.params.params}
        assert params == {"period": "Q1", "depth": "detailed"}

    def test_with_constraints(self):
        node = parse("@CFO ANALYZE revenue [margin > 20%]")
        assert isinstance(node, DirectiveNode)
        assert len(node.constraints.constraints) == 1
        assert "margin" in node.constraints.constraints[0].text

    def test_with_priority(self):
        node = parse("@CEO DECIDE proposal !urgent")
        assert node.priority.level == "urgent"

    def test_with_modifiers(self):
        node = parse("@CTO ASSESS infra ~thorough ~discretion")
        assert [m.name for m in node.modifiers] == ["thorough", "discretion"]

    def test_with_format_hint(self):
        node = parse("@CFO ANALYZE revenue -> summary")
        assert node.output is not None
        assert node.output.format == "summary"

    def test_with_contract(self):
        node = parse("@CTO ANALYZE proposal -> {confidence: float, recommendation: str}")
        assert node.response_contract is not None
        assert len(node.response_contract.fields) == 2
        assert node.response_contract.fields[0] == ContractFieldNode("confidence", "float")
        assert node.response_contract.fields[1] == ContractFieldNode("recommendation", "str")

    def test_optional_contract_field(self):
        node = parse("@CTO ANALYZE x -> {score: float, notes?: str}")
        assert node.response_contract.fields[1].required is False

    def test_with_bb_refs(self):
        node = parse("@CFO ANALYZE revenue [bb:financial:revenue@v2]")
        assert len(node.context_refs) == 1
        assert node.context_refs[0].namespace == "financial"
        assert node.context_refs[0].key == "revenue"
        assert node.context_refs[0].version == 2

    def test_prev_ref_target(self):
        node = parse("@CFO ANALYZE $prev")
        assert isinstance(node.target, PrevRefNode)

    def test_bb_ref_target(self):
        node = parse("@CFO ANALYZE bb:org:state")
        assert isinstance(node.target, BlackboardRefNode)
        assert node.target.namespace == "org"

    def test_number_param(self):
        node = parse("@CFO FORECAST revenue {horizon=3, threshold=1M}")
        params = {p.key: p.value for p in node.params.params}
        assert params["horizon"] == "3"
        assert params["threshold"] == "1M"

    def test_string_param(self):
        node = parse('@CEO DECIDE expansion {reason="market growth"}')
        params = {p.key: p.value for p in node.params.params}
        assert params["reason"] == "market growth"

    def test_full_directive(self):
        src = "@CFO ANALYZE revenue {period=Q1} [margin > 20%] -> summary !urgent ~thorough"
        node = parse(src)
        assert node.recipient.agent_code == "CFO"
        assert node.action == "ANALYZE"
        assert node.target == "revenue"
        params = {p.key: p.value for p in node.params.params}
        assert params == {"period": "Q1"}
        assert len(node.constraints.constraints) == 1
        assert node.output.format == "summary"
        assert node.priority.level == "urgent"
        assert [m.name for m in node.modifiers] == ["thorough"]


class TestRetry:
    def test_basic_retry(self):
        node = parse("@CTO ANALYZE infra RETRY 3")
        assert node.retry is not None
        assert node.retry.max_retries == 3
        assert node.retry.backoff == "exp"

    def test_retry_with_backoff(self):
        node = parse("@CTO ANALYZE infra RETRY 5 BACKOFF linear")
        assert node.retry.backoff == "linear"

    def test_retry_with_fallback(self):
        node = parse("@CTO ANALYZE infra RETRY 3 FALLBACK @CIO")
        assert node.retry.fallback_agent == "CIO"

    def test_full_retry(self):
        node = parse("@CTO ANALYZE infra RETRY 3 BACKOFF exp FALLBACK @CIO")
        assert node.retry.max_retries == 3
        assert node.retry.backoff == "exp"
        assert node.retry.fallback_agent == "CIO"


class TestPipeline:
    def test_two_step(self):
        node = parse("@CDO GATHER data | @CFO ANALYZE $prev")
        assert isinstance(node, PipelineNode)
        assert len(node.directives) == 2
        assert node.directives[0].recipient.agent_code == "CDO"
        assert node.directives[1].recipient.agent_code == "CFO"
        assert isinstance(node.directives[1].target, PrevRefNode)

    def test_three_step(self):
        node = parse("@CDO GATHER metrics | @CFO ANALYZE $prev | @CEO DECIDE $prev")
        assert isinstance(node, PipelineNode)
        assert len(node.directives) == 3

    def test_pipeline_with_params(self):
        node = parse("@CDO GATHER data {source=api} | @CFO ANALYZE $prev {depth=detailed}")
        assert isinstance(node, PipelineNode)
        p0 = {p.key: p.value for p in node.directives[0].params.params}
        p1 = {p.key: p.value for p in node.directives[1].params.params}
        assert p0 == {"source": "api"}
        assert p1 == {"depth": "detailed"}


class TestParallelBlock:
    def test_two_branches(self):
        node = parse("PAR { @CFO FORECAST revenue; @CTO ASSESS infra }")
        assert isinstance(node, ParallelBlockNode)
        assert len(node.branches) == 2
        assert node.branches[0].recipient.agent_code == "CFO"
        assert node.branches[1].recipient.agent_code == "CTO"

    def test_three_branches(self):
        node = parse("PAR { @CFO FORECAST revenue; @CTO ASSESS infra; @CMO ANALYZE market }")
        assert isinstance(node, ParallelBlockNode)
        assert len(node.branches) == 3

    def test_nested_pipeline_in_par(self):
        src = "PAR { @CDO GATHER data | @CFO ANALYZE $prev; @CTO ASSESS infra }"
        node = parse(src)
        assert isinstance(node, ParallelBlockNode)
        assert isinstance(node.branches[0], PipelineNode)
        assert isinstance(node.branches[1], DirectiveNode)


class TestSequentialBlock:
    def test_basic(self):
        node = parse("SEQ { @CDO GATHER data; @CFO ANALYZE $prev }")
        assert isinstance(node, SequentialBlockNode)
        assert len(node.steps) == 2

    def test_three_steps(self):
        node = parse("SEQ { @CDO GATHER data; @CFO ANALYZE $prev; @CEO DECIDE $prev }")
        assert isinstance(node, SequentialBlockNode)
        assert len(node.steps) == 3


class TestConditional:
    def test_if_then(self):
        node = parse("IF @CRO.pipeline > 1M THEN @CEO DECIDE expansion")
        assert isinstance(node, ConditionalNode)
        condition_str = str(node.condition)
        assert "CRO" in condition_str or "pipeline" in condition_str
        assert isinstance(node.then_branch, DirectiveNode)
        assert node.then_branch.recipient.agent_code == "CEO"
        assert node.else_branch is None

    def test_if_then_else(self):
        node = parse("IF @CRO.pipeline > 1M THEN @CEO DECIDE expand ELSE @CFO ANALYZE cuts")
        assert isinstance(node, ConditionalNode)
        assert isinstance(node.then_branch, DirectiveNode)
        assert isinstance(node.else_branch, DirectiveNode)
        assert node.else_branch.recipient.agent_code == "CFO"


class TestProgram:
    def test_multi_statement_newline(self):
        src = "@CEO DECIDE policy\n@CFO ANALYZE revenue"
        node = parse(src)
        assert isinstance(node, ProgramNode)
        assert len(node.statements) == 2

    def test_multi_statement_semicolon(self):
        src = "@CEO DECIDE policy; @CFO ANALYZE revenue"
        node = parse(src)
        assert isinstance(node, ProgramNode)
        assert len(node.statements) == 2

    def test_single_statement_no_program(self):
        node = parse("@CEO DECIDE expansion")
        assert isinstance(node, DirectiveNode)  # Not wrapped in Program


class TestEdgeCases:
    def test_empty_params(self):
        node = parse("@CEO DECIDE expansion {}")
        assert node.params is None or len(node.params.params) == 0

    def test_trailing_semicolon(self):
        node = parse("@CEO DECIDE expansion;")
        assert isinstance(node, DirectiveNode)

    def test_comments_ignored(self):
        src = "# This is a comment\n@CEO DECIDE expansion"
        node = parse(src)
        assert isinstance(node, DirectiveNode)

    def test_multiple_bb_refs(self):
        node = parse("@CFO ANALYZE revenue [bb:financial:revenue@v1, bb:org:policy@v2]")
        assert len(node.context_refs) == 2

    def test_agent_field_in_condition(self):
        node = parse("IF @CRO.pipeline > 1M THEN @CEO DECIDE go")
        condition_str = str(node.condition)
        assert "CRO" in condition_str or "pipeline" in condition_str
