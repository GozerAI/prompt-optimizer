"""Extended lexer tests for orchestration tokens (ported from AIL)."""

import pytest
from prompt_optimizer.grammar.lexer import Lexer
from prompt_optimizer.grammar.tokens import TokenType


def tokenize(src: str) -> list:
    return Lexer().tokenize(src)


class TestBasicTokens:
    def test_agent_ref(self):
        tokens = tokenize("@CEO")
        # prompt-optimizer emits AT + AGENT_CODE
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.AT in types
        assert TokenType.AGENT_CODE in types

    def test_agent_field_ref(self):
        tokens = tokenize("@CRO.pipeline")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.AGENT_FIELD in types

    def test_prev_ref(self):
        tokens = tokenize("$prev")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.PREV_REF in types

    def test_prev_ref_indexed(self):
        tokens = tokenize("$prev[2]")
        prev_tokens = [t for t in tokens if t.type == TokenType.PREV_REF]
        assert len(prev_tokens) == 1
        assert "2" in prev_tokens[0].value

    def test_bb_ref(self):
        tokens = tokenize("bb:org:state@v2")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.BB_REF in types

    def test_bb_ref_no_version(self):
        tokens = tokenize("bb:financial:revenue")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.BB_REF in types

    def test_action(self):
        tokens = tokenize("ANALYZE")
        action_tokens = [t for t in tokens if t.type == TokenType.ACTION]
        assert len(action_tokens) == 1
        assert action_tokens[0].value == "ANALYZE"

    def test_priority(self):
        tokens = tokenize("!urgent")
        priority_tokens = [t for t in tokens if t.type == TokenType.PRIORITY]
        assert len(priority_tokens) == 1

    def test_modifier(self):
        tokens = tokenize("~thorough")
        mod_tokens = [t for t in tokens if t.type == TokenType.MODIFIER]
        assert len(mod_tokens) == 1

    def test_number(self):
        tokens = tokenize("42")
        assert any(t.type == TokenType.NUMBER for t in tokens)

    def test_number_with_suffix(self):
        tokens = tokenize("1M")
        num_tokens = [t for t in tokens if t.type == TokenType.NUMBER]
        assert len(num_tokens) == 1
        assert num_tokens[0].value == "1M"

    def test_string_double(self):
        tokens = tokenize('"hello world"')
        assert any(t.type == TokenType.STRING for t in tokens)

    def test_string_single(self):
        tokens = tokenize("'hello'")
        assert any(t.type == TokenType.STRING for t in tokens)

    def test_arrow(self):
        tokens = tokenize("->")
        assert any(t.type == TokenType.ARROW for t in tokens)

    def test_pipe(self):
        tokens = tokenize("|")
        assert any(t.type == TokenType.PIPE for t in tokens)

    def test_keywords(self):
        for kw in ["PAR", "SEQ", "IF", "THEN", "ELSE", "RETRY", "BACKOFF", "FALLBACK"]:
            tokens = tokenize(kw)
            types = [t.type for t in tokens if t.type != TokenType.EOF]
            assert len(types) >= 1
            assert types[0].name == kw

    def test_ident(self):
        tokens = tokenize("revenue")
        assert any(t.type == TokenType.IDENTIFIER for t in tokens)

    def test_comment_skipped(self):
        tokens = tokenize("# this is a comment\n@CEO")
        # First meaningful token should be AT or NEWLINE
        agents = [t for t in tokens if t.type == TokenType.AGENT_CODE]
        assert len(agents) == 1


class TestFullDirective:
    def test_simple_directive_tokens(self):
        src = "@CFO ANALYZE revenue"
        tokens = tokenize(src)
        types = [t.type for t in tokens if t.type not in (TokenType.EOF, TokenType.NEWLINE)]
        assert TokenType.AT in types
        assert TokenType.AGENT_CODE in types
        assert TokenType.ACTION in types
        assert TokenType.IDENTIFIER in types

    def test_directive_with_params(self):
        src = "@CFO ANALYZE revenue {period=Q1, depth=detailed}"
        tokens = tokenize(src)
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.LBRACE in types
        assert TokenType.RBRACE in types
        assert TokenType.EQUALS in types

    def test_directive_with_priority_and_modifiers(self):
        src = "@CTO ASSESS infra !urgent ~thorough"
        tokens = tokenize(src)
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.PRIORITY in types
        assert TokenType.MODIFIER in types

    def test_pipeline_tokens(self):
        src = "@CDO GATHER data | @CFO ANALYZE $prev"
        tokens = tokenize(src)
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert types.count(TokenType.PIPE) == 1
        assert types.count(TokenType.AGENT_CODE) == 2

    def test_par_block_tokens(self):
        src = "PAR { @CFO FORECAST revenue; @CTO ASSESS infra }"
        tokens = tokenize(src)
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.PAR in types
        assert TokenType.LBRACE in types
        assert TokenType.SEMICOLON in types
        assert TokenType.RBRACE in types
