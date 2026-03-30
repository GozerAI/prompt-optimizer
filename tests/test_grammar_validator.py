"""Tests for the grammar validator (ported from AIL)."""

import pytest
from prompt_optimizer.grammar.lexer import Lexer
from prompt_optimizer.grammar.parser import Parser
from prompt_optimizer.grammar.validator import Validator


def validate(src: str, known_agents=None):
    tokens = Lexer().tokenize(src)
    node = Parser(tokens).parse()
    return Validator(known_agents=known_agents).validate(node)


class TestValidDirectives:
    def test_simple_valid(self):
        result = validate("@CEO DECIDE expansion")
        assert result.valid

    def test_pipeline_valid(self):
        result = validate("@CDO GATHER data | @CFO ANALYZE $prev")
        assert result.valid

    def test_par_block_valid(self):
        result = validate("PAR { @CFO FORECAST revenue; @CTO ASSESS infra }")
        assert result.valid

    def test_seq_block_valid(self):
        result = validate("SEQ { @CDO GATHER data; @CFO ANALYZE $prev }")
        assert result.valid


class TestPrevRefValidation:
    def test_prev_in_first_step_errors(self):
        result = validate("@CFO ANALYZE $prev")
        assert not result.valid
        assert any("$prev" in e for e in result.errors)

    def test_prev_in_pipeline_ok(self):
        result = validate("@CDO GATHER data | @CFO ANALYZE $prev")
        assert result.valid

    def test_prev_in_par_warns(self):
        result = validate("PAR { @CFO ANALYZE $prev; @CTO ASSESS infra }")
        # $prev in PAR is ambiguous — should warn
        assert len(result.warnings) > 0


class TestAgentValidation:
    def test_unknown_agent_warns(self):
        known = {"CEO", "CFO", "CTO"}
        result = validate("@XYZ DECIDE something", known_agents=known)
        assert any("XYZ" in w for w in result.warnings)

    def test_known_agent_no_warning(self):
        known = {"CEO", "CFO", "CTO"}
        result = validate("@CEO DECIDE something", known_agents=known)
        assert len(result.warnings) == 0


class TestConditionalValidation:
    def test_conditional_valid(self):
        result = validate("IF @CRO.pipeline > 1M THEN @CEO DECIDE expand")
        assert result.valid
