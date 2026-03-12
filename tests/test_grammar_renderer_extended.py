"""Extended renderer roundtrip tests (ported from AIL emitter tests)."""

import pytest
from prompt_optimizer.grammar.lexer import Lexer
from prompt_optimizer.grammar.parser import Parser
from prompt_optimizer.grammar.renderer import Renderer


def roundtrip(src: str) -> str:
    """Parse then render — should produce equivalent output."""
    tokens = Lexer().tokenize(src)
    node = Parser(tokens).parse()
    return Renderer().render(node)


class TestRoundtrip:
    def test_simple_directive(self):
        assert roundtrip("@CEO DECIDE expansion") == "@CEO DECIDE expansion"

    def test_with_params(self):
        result = roundtrip("@CFO ANALYZE revenue {period=Q1}")
        assert "@CFO ANALYZE revenue {period=Q1}" == result

    def test_with_priority(self):
        result = roundtrip("@CEO DECIDE proposal !urgent")
        assert "!urgent" in result

    def test_with_modifier(self):
        result = roundtrip("@CTO ASSESS infra ~thorough")
        assert "~thorough" in result

    def test_format_hint(self):
        result = roundtrip("@CFO ANALYZE revenue -> summary")
        assert "-> summary" in result

    def test_contract(self):
        result = roundtrip("@CTO ANALYZE x -> {score: float, note: str}")
        assert "score: float" in result
        assert "note: str" in result

    def test_pipeline(self):
        result = roundtrip("@CDO GATHER data | @CFO ANALYZE $prev")
        assert " | " in result
        assert "@CDO GATHER data" in result
        assert "@CFO ANALYZE $prev" in result

    def test_par_block(self):
        result = roundtrip("PAR { @CFO FORECAST revenue; @CTO ASSESS infra }")
        assert result.startswith("PAR {")
        assert "@CFO FORECAST revenue" in result
        assert "@CTO ASSESS infra" in result

    def test_seq_block(self):
        result = roundtrip("SEQ { @CDO GATHER data; @CFO ANALYZE $prev }")
        assert result.startswith("SEQ {")

    def test_conditional(self):
        src = "IF @CRO.pipeline > 1M THEN @CEO DECIDE expand ELSE @CFO ANALYZE cuts"
        result = roundtrip(src)
        assert "IF" in result
        assert "THEN" in result
        assert "ELSE" in result

    def test_retry(self):
        result = roundtrip("@CTO ANALYZE infra RETRY 3 FALLBACK @CIO")
        assert "RETRY 3" in result
        assert "FALLBACK @CIO" in result

    def test_bb_refs(self):
        result = roundtrip("@CFO ANALYZE revenue [bb:financial:revenue@v2]")
        assert "bb:financial:revenue@v2" in result

    def test_constraints(self):
        result = roundtrip("@CFO ANALYZE revenue [margin > 20%]")
        assert "margin > 20%" in result
