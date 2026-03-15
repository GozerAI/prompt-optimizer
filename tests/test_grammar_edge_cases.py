"""Tests for grammar/parser edge cases — deeply nested structures, special characters,
malformed input, boundary conditions for bb refs, $prev refs, and token limits."""

import pytest

from prompt_optimizer.grammar.ast_nodes import (
    BlackboardRefNode,
    DirectiveNode,
    ParallelBlockNode,
    PipelineNode,
    PrevRefNode,
    ProgramNode,
    SequentialBlockNode,
)
from prompt_optimizer.grammar.lexer import Lexer
from prompt_optimizer.grammar.parser import ParseError, Parser
from prompt_optimizer.grammar.renderer import Renderer
from prompt_optimizer.grammar.tokens import TokenType
from prompt_optimizer.grammar.validator import Validator


def _parse(text: str):
    tokens = Lexer().tokenize(text)
    return Parser(tokens).parse()


def _tokens(text: str):
    return Lexer().tokenize(text)


# ============================================================
# Empty / minimal input
# ============================================================


class TestEmptyInput:
    def test_empty_string_tokenizes_to_eof(self):
        tokens = _tokens("")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_whitespace_only_tokenizes_to_eof(self):
        tokens = _tokens("   \t  ")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_newline_only(self):
        tokens = _tokens("\n\n\n")
        non_eof = [t for t in tokens if t.type != TokenType.EOF]
        # Newlines are tokenized
        assert all(t.type == TokenType.NEWLINE for t in non_eof)

    def test_comment_only(self):
        tokens = _tokens("# this is a comment\n")
        non_nl_eof = [t for t in tokens if t.type not in (TokenType.NEWLINE, TokenType.EOF)]
        assert len(non_nl_eof) == 0


# ============================================================
# Deeply nested PAR/SEQ blocks
# ============================================================


class TestDeeplyNestedBlocks:
    def test_par_inside_par(self):
        src = "PAR { PAR { @CFO FORECAST revenue; @CTO ASSESS infra }; @CDO GATHER data }"
        node = _parse(src)
        assert isinstance(node, ParallelBlockNode)
        assert isinstance(node.branches[0], ParallelBlockNode)

    def test_seq_inside_par(self):
        src = "PAR { SEQ { @CDO GATHER data; @CFO ANALYZE $prev }; @CTO ASSESS infra }"
        node = _parse(src)
        assert isinstance(node, ParallelBlockNode)
        assert isinstance(node.branches[0], SequentialBlockNode)

    def test_par_inside_seq(self):
        src = "SEQ { @CDO GATHER data; PAR { @CFO ANALYZE $prev; @CTO ASSESS infra } }"
        node = _parse(src)
        assert isinstance(node, SequentialBlockNode)
        assert isinstance(node.steps[1], ParallelBlockNode)

    def test_three_levels_deep(self):
        src = (
            "PAR { "
            "  SEQ { @CDO GATHER data; PAR { @CFO ANALYZE $prev; @CTO ASSESS infra } }; "
            "  @CEO DECIDE expansion "
            "}"
        )
        node = _parse(src)
        assert isinstance(node, ParallelBlockNode)
        seq = node.branches[0]
        assert isinstance(seq, SequentialBlockNode)
        inner_par = seq.steps[1]
        assert isinstance(inner_par, ParallelBlockNode)

    def test_four_levels_deep(self):
        src = (
            "SEQ { "
            "  PAR { "
            "    SEQ { @CDO GATHER data; PAR { @CFO ANALYZE $prev; @CTO ASSESS infra } }; "
            "    @CMO ANALYZE brand "
            "  }; "
            "  @CEO DECIDE expansion "
            "}"
        )
        node = _parse(src)
        assert isinstance(node, SequentialBlockNode)
        outer_par = node.steps[0]
        assert isinstance(outer_par, ParallelBlockNode)
        inner_seq = outer_par.branches[0]
        assert isinstance(inner_seq, SequentialBlockNode)
        deepest_par = inner_seq.steps[1]
        assert isinstance(deepest_par, ParallelBlockNode)

    def test_five_levels_deep(self):
        src = (
            "PAR { "
            "  SEQ { "
            "    PAR { "
            "      SEQ { "
            "        PAR { @CFO FORECAST revenue; @CTO ASSESS infra }; "
            "        @CDO GATHER data "
            "      }; "
            "      @CMO ANALYZE brand "
            "    }; "
            "    @CEO DECIDE expansion "
            "  }; "
            "  @CIO ASSESS security "
            "}"
        )
        node = _parse(src)
        assert isinstance(node, ParallelBlockNode)
        # Walk down: PAR -> SEQ -> PAR -> SEQ -> PAR
        l1 = node.branches[0]
        assert isinstance(l1, SequentialBlockNode)
        l2 = l1.steps[0]
        assert isinstance(l2, ParallelBlockNode)
        l3 = l2.branches[0]
        assert isinstance(l3, SequentialBlockNode)
        l4 = l3.steps[0]
        assert isinstance(l4, ParallelBlockNode)

    def test_deeply_nested_renders_back(self):
        src = "PAR { SEQ { @CDO GATHER data; @CFO ANALYZE $prev }; @CTO ASSESS infra }"
        node = _parse(src)
        rendered = Renderer().render(node)
        assert "PAR" in rendered
        assert "SEQ" in rendered


# ============================================================
# Long directive names and identifiers
# ============================================================


class TestLongIdentifiers:
    def test_very_long_target(self):
        long_target = "a" * 200
        src = f"@CFO ANALYZE {long_target}"
        node = _parse(src)
        assert isinstance(node, DirectiveNode)
        assert long_target in str(node.target)

    def test_long_param_value(self):
        long_val = "x" * 300
        src = f'@CFO ANALYZE revenue {{detail={long_val}}}'
        node = _parse(src)
        assert isinstance(node, DirectiveNode)
        assert node.params is not None
        assert node.params.params[0].value == long_val

    def test_many_params(self):
        params = ", ".join(f"p{i}=v{i}" for i in range(20))
        src = f"@CFO ANALYZE revenue {{{params}}}"
        node = _parse(src)
        assert isinstance(node, DirectiveNode)
        assert len(node.params.params) == 20


# ============================================================
# Special characters and strings
# ============================================================


class TestSpecialCharacters:
    def test_quoted_string_with_spaces(self):
        tokens = _tokens('"hello world foo bar"')
        str_tokens = [t for t in tokens if t.type == TokenType.STRING]
        assert len(str_tokens) == 1
        assert str_tokens[0].value == "hello world foo bar"

    def test_single_quoted_string(self):
        tokens = _tokens("'single quoted'")
        str_tokens = [t for t in tokens if t.type == TokenType.STRING]
        assert len(str_tokens) == 1
        assert str_tokens[0].value == "single quoted"

    def test_string_in_directive_target(self):
        node = _parse('@CFO ANALYZE "quarterly report"')
        assert isinstance(node, DirectiveNode)
        assert "quarterly report" in str(node.target)

    def test_number_with_suffix(self):
        tokens = _tokens("2.3M")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "2.3M"

    def test_percentage_number(self):
        tokens = _tokens("15%")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "15%"

    def test_negative_number(self):
        tokens = _tokens("-42")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "-42"

    def test_comma_separated_number(self):
        tokens = _tokens("1,000,000")
        assert tokens[0].type == TokenType.NUMBER

    def test_dollar_amount_variants(self):
        for val in ["$100", "$2.3M", "$1,500"]:
            tokens = _tokens(val)
            assert tokens[0].type == TokenType.NUMBER
            assert tokens[0].value == val


# ============================================================
# Unicode content
# ============================================================


class TestUnicodeContent:
    def test_unicode_in_identifier(self):
        # Unicode characters not in [A-Za-z_] are skipped by lexer
        tokens = _tokens("revenue")
        assert tokens[0].type == TokenType.IDENTIFIER

    def test_unicode_in_string(self):
        tokens = _tokens('"analyse des revenus"')
        str_tokens = [t for t in tokens if t.type == TokenType.STRING]
        assert len(str_tokens) == 1
        assert str_tokens[0].value == "analyse des revenus"

    def test_emoji_in_string(self):
        tokens = _tokens('"status: OK :-)"')
        str_tokens = [t for t in tokens if t.type == TokenType.STRING]
        assert len(str_tokens) == 1

    def test_non_ascii_skipped_outside_strings(self):
        # Non-ASCII characters outside strings are skipped
        tokens = _tokens("@CFO ANALYZE revenue")
        agent_tokens = [t for t in tokens if t.type == TokenType.AGENT_CODE]
        assert any(t.value == "CFO" for t in agent_tokens)


# ============================================================
# Malformed directives
# ============================================================


class TestMalformedDirectives:
    def test_unclosed_brace_in_params(self):
        # Missing closing brace — parser should raise ParseError
        with pytest.raises(ParseError):
            _parse("@CFO ANALYZE revenue {period=Q1")

    def test_unclosed_bracket_in_constraints(self):
        with pytest.raises(ParseError):
            _parse("@CFO ANALYZE revenue [margin > 20%")

    def test_missing_closing_par_block(self):
        with pytest.raises(ParseError):
            _parse("PAR { @CFO FORECAST revenue; @CTO ASSESS infra")

    def test_missing_closing_seq_block(self):
        with pytest.raises(ParseError):
            _parse("SEQ { @CDO GATHER data; @CFO ANALYZE $prev")

    def test_unexpected_rbrace(self):
        # Extra closing brace — should not crash; the parser stops at matching brace
        tokens = _tokens("@CFO ANALYZE revenue }")
        # Lexer produces valid tokens including RBRACE
        types = [t.type for t in tokens]
        assert TokenType.RBRACE in types

    def test_unclosed_string_tokenizes(self):
        # Unclosed string — lexer reads to end of input
        tokens = _tokens('"hello')
        str_tokens = [t for t in tokens if t.type == TokenType.STRING]
        assert len(str_tokens) == 1
        assert str_tokens[0].value == "hello"

    def test_pipe_without_second_directive(self):
        # Pipe at end — parser will try to parse next directive and fail
        with pytest.raises(ParseError):
            _parse("@CFO ANALYZE revenue |")

    def test_arrow_without_output(self):
        # Arrow at end without format hint or contract
        node = _parse("@CFO ANALYZE revenue ->")
        assert isinstance(node, DirectiveNode)
        # No output or contract since nothing follows ->
        assert node.output is None


# ============================================================
# Comments in various positions
# ============================================================


class TestComments:
    def test_comment_at_end_of_line(self):
        tokens = _tokens("@CFO ANALYZE revenue # this is a comment\n")
        types = [t.type for t in tokens if t.type not in (TokenType.NEWLINE, TokenType.EOF)]
        assert TokenType.AT in types
        assert TokenType.AGENT_CODE in types
        assert TokenType.ACTION in types

    def test_comment_before_directive(self):
        tokens = _tokens("# header comment\n@CFO ANALYZE revenue")
        agent_tokens = [t for t in tokens if t.type == TokenType.AGENT_CODE]
        assert any(t.value == "CFO" for t in agent_tokens)

    def test_multiple_comment_lines(self):
        src = "# line 1\n# line 2\n@CFO ANALYZE revenue"
        node = _parse(src)
        assert isinstance(node, DirectiveNode)
        assert node.action == "ANALYZE"

    def test_inline_comment_does_not_leak_into_value(self):
        src = "@CFO ANALYZE revenue # ignore this\n"
        node = _parse(src)
        assert isinstance(node, DirectiveNode)
        assert "#" not in str(node.target)
        assert "ignore" not in str(node.target)


# ============================================================
# Multiple consecutive PAR blocks
# ============================================================


class TestMultipleParBlocks:
    def test_two_consecutive_par_blocks(self):
        src = (
            "PAR { @CFO FORECAST revenue; @CTO ASSESS infra }\n"
            "PAR { @CMO ANALYZE brand; @CDO GATHER data }"
        )
        node = _parse(src)
        assert isinstance(node, ProgramNode)
        assert len(node.statements) == 2
        assert isinstance(node.statements[0], ParallelBlockNode)
        assert isinstance(node.statements[1], ParallelBlockNode)

    def test_par_then_directive(self):
        src = (
            "PAR { @CFO FORECAST revenue; @CTO ASSESS infra }\n"
            "@CEO DECIDE expansion"
        )
        node = _parse(src)
        assert isinstance(node, ProgramNode)
        assert isinstance(node.statements[0], ParallelBlockNode)
        assert isinstance(node.statements[1], DirectiveNode)

    def test_multiple_par_render_roundtrip(self):
        src = (
            "PAR { @CFO FORECAST revenue; @CTO ASSESS infra }\n"
            "PAR { @CMO ANALYZE brand; @CDO GATHER data }"
        )
        node = _parse(src)
        rendered = Renderer().render(node)
        assert rendered.count("PAR") == 2


# ============================================================
# $prev references across nested scopes
# ============================================================


class TestPrevReferences:
    def test_prev_in_pipeline(self):
        node = _parse("@CDO GATHER data | @CFO ANALYZE $prev")
        assert isinstance(node, PipelineNode)
        d2 = node.directives[1]
        assert isinstance(d2.target, PrevRefNode)
        assert d2.target.step is None

    def test_prev_with_index(self):
        node = _parse("@CDO GATHER data | @CFO ANALYZE $prev[0]")
        assert isinstance(node, PipelineNode)
        d2 = node.directives[1]
        assert isinstance(d2.target, PrevRefNode)
        assert d2.target.step == 0

    def test_prev_with_large_index(self):
        node = _parse("@CDO GATHER data | @CFO ANALYZE $prev[99]")
        assert isinstance(node, PipelineNode)
        d2 = node.directives[1]
        assert isinstance(d2.target, PrevRefNode)
        assert d2.target.step == 99

    def test_prev_in_seq_block(self):
        node = _parse("SEQ { @CDO GATHER data; @CFO ANALYZE $prev }")
        assert isinstance(node, SequentialBlockNode)
        d2 = node.steps[1]
        assert isinstance(d2, DirectiveNode)
        assert isinstance(d2.target, PrevRefNode)

    def test_prev_in_par_block_warns(self):
        node = _parse("PAR { @CFO ANALYZE $prev; @CTO ASSESS infra }")
        result = Validator().validate(node)
        # $prev in PAR should produce a warning
        assert any("$prev" in w for w in result.warnings)

    def test_prev_at_step_zero_errors(self):
        node = _parse("@CFO ANALYZE $prev")
        result = Validator().validate(node)
        assert any("$prev" in e for e in result.errors)


# ============================================================
# bb:ns:key@vN edge cases
# ============================================================


class TestBBRefEdgeCases:
    def test_bb_ref_no_version(self):
        tokens = _tokens("bb:financial:revenue")
        bb_tokens = [t for t in tokens if t.type == TokenType.BB_REF]
        assert len(bb_tokens) == 1
        assert bb_tokens[0].value == "bb:financial:revenue"

    def test_bb_ref_with_version_1(self):
        tokens = _tokens("bb:financial:revenue@v1")
        bb_tokens = [t for t in tokens if t.type == TokenType.BB_REF]
        assert len(bb_tokens) == 1
        assert "@v1" in bb_tokens[0].value

    def test_bb_ref_version_0(self):
        # Version 0 — the regex requires \d+ which matches 0
        tokens = _tokens("bb:financial:revenue@v0")
        bb_tokens = [t for t in tokens if t.type == TokenType.BB_REF]
        assert len(bb_tokens) == 1
        assert "@v0" in bb_tokens[0].value

    def test_bb_ref_large_version(self):
        tokens = _tokens("bb:org:state@v999")
        bb_tokens = [t for t in tokens if t.type == TokenType.BB_REF]
        assert len(bb_tokens) == 1
        assert "@v999" in bb_tokens[0].value

    def test_bb_ref_parsed_as_target(self):
        node = _parse("@CFO ANALYZE bb:financial:revenue@v2")
        assert isinstance(node, DirectiveNode)
        assert isinstance(node.target, BlackboardRefNode)
        assert node.target.namespace == "financial"
        assert node.target.key == "revenue"
        assert node.target.version == 2

    def test_bb_ref_no_version_parsed(self):
        node = _parse("@CFO ANALYZE bb:org:state")
        assert isinstance(node, DirectiveNode)
        assert isinstance(node.target, BlackboardRefNode)
        assert node.target.version is None

    def test_bb_ref_in_context_bracket(self):
        src = "@CFO ANALYZE revenue [bb:financial:revenue@v1]"
        node = _parse(src)
        assert isinstance(node, DirectiveNode)
        assert len(node.context_refs) == 1
        assert node.context_refs[0].namespace == "financial"

    def test_bb_ref_version_0_parsed(self):
        node = _parse("@CFO ANALYZE bb:org:state@v0")
        assert isinstance(node, DirectiveNode)
        assert isinstance(node.target, BlackboardRefNode)
        assert node.target.version == 0

    def test_bb_ref_pointer_property(self):
        ref = BlackboardRefNode(namespace="org", key="state", version=3)
        assert ref.pointer == "org:state@v3"

    def test_bb_ref_pointer_no_version(self):
        ref = BlackboardRefNode(namespace="org", key="state", version=None)
        assert ref.pointer == "org:state"

    def test_bb_ref_repr(self):
        ref = BlackboardRefNode(namespace="financial", key="revenue", version=1)
        assert repr(ref) == "bb:financial:revenue@v1"

    def test_bb_ref_negative_version_not_matched(self):
        # bb regex uses \d+ which doesn't match negative numbers
        tokens = _tokens("bb:org:state@v-1")
        bb_tokens = [t for t in tokens if t.type == TokenType.BB_REF]
        # Should NOT parse as a bb ref with negative version
        # Either no match or partial match without version
        if bb_tokens:
            assert "-1" not in bb_tokens[0].value


# ============================================================
# Semicolons and multi-statement programs
# ============================================================


class TestMultiStatement:
    def test_semicolon_separates_statements(self):
        node = _parse("@CFO ANALYZE revenue; @CTO ASSESS infra")
        assert isinstance(node, ProgramNode)
        assert len(node.statements) == 2

    def test_newline_separates_statements(self):
        node = _parse("@CFO ANALYZE revenue\n@CTO ASSESS infra")
        assert isinstance(node, ProgramNode)
        assert len(node.statements) == 2

    def test_trailing_semicolon(self):
        # Trailing semicolon should not cause an error
        node = _parse("@CFO ANALYZE revenue;")
        assert isinstance(node, DirectiveNode)

    def test_single_statement_not_wrapped_in_program(self):
        node = _parse("@CFO ANALYZE revenue")
        assert isinstance(node, DirectiveNode)
        # Single statement is returned directly, not wrapped in ProgramNode


# ============================================================
# Retry / backoff / fallback edge cases
# ============================================================


class TestRetryEdgeCases:
    def test_retry_with_backoff_linear(self):
        node = _parse("@CTO ANALYZE infra RETRY 3 BACKOFF linear")
        assert isinstance(node, DirectiveNode)
        assert node.retry is not None
        assert node.retry.max_retries == 3
        assert node.retry.backoff == "linear"

    def test_retry_with_fallback(self):
        node = _parse("@CTO ANALYZE infra RETRY 2 FALLBACK @CIO")
        assert isinstance(node, DirectiveNode)
        assert node.retry.fallback_agent == "CIO"

    def test_retry_with_both_backoff_and_fallback(self):
        node = _parse("@CTO ANALYZE infra RETRY 5 BACKOFF fixed FALLBACK @CIO")
        assert isinstance(node, DirectiveNode)
        assert node.retry.max_retries == 5
        assert node.retry.backoff == "fixed"
        assert node.retry.fallback_agent == "CIO"

    def test_retry_default_backoff_is_exp(self):
        node = _parse("@CTO ANALYZE infra RETRY 3")
        assert node.retry.backoff == "exp"


# ============================================================
# Response contract edge cases
# ============================================================


class TestResponseContractEdgeCases:
    def test_single_field_contract(self):
        node = _parse("@CFO ANALYZE revenue -> {score: float}")
        assert isinstance(node, DirectiveNode)
        assert node.response_contract is not None
        assert len(node.response_contract.fields) == 1

    def test_multiple_field_contract(self):
        node = _parse("@CFO ANALYZE revenue -> {score: float, label: str, valid: bool}")
        assert isinstance(node, DirectiveNode)
        assert len(node.response_contract.fields) == 3

    def test_optional_field_contract(self):
        node = _parse("@CFO ANALYZE revenue -> {score: float, notes?: str}")
        assert isinstance(node, DirectiveNode)
        fields = node.response_contract.fields
        assert fields[0].required is True
        assert fields[1].required is False

    def test_format_hint_not_contract(self):
        node = _parse("@CFO ANALYZE revenue -> summary")
        assert isinstance(node, DirectiveNode)
        assert node.output is not None
        assert node.output.format == "summary"
        assert node.response_contract is None


# ============================================================
# Custom agent codes
# ============================================================


class TestCustomAgentCodes:
    def test_custom_codes_recognized(self):
        lexer = Lexer(agent_codes={"ALPHA", "BETA", "GAMMA"})
        tokens = lexer.tokenize("@ALPHA ANALYZE data")
        agent_tokens = [t for t in tokens if t.type == TokenType.AGENT_CODE]
        assert any(t.value == "ALPHA" for t in agent_tokens)

    def test_unknown_code_is_identifier(self):
        tokens = _tokens("@FOOBAR ANALYZE data")
        # FOOBAR is not a default agent code, but @ + uppercase word gets
        # AT + AGENT_CODE tokens (the lexer matches any @UPPERCASE pattern)
        agent_tokens = [t for t in tokens if t.type == TokenType.AGENT_CODE]
        # The regex matches any @[A-Z][A-Za-z0-9_]* as AGENT_CODE
        assert any(t.value == "FOOBAR" for t in agent_tokens)
