"""Tests for Layer 1: Structural compression."""


from prompt_optimizer.layers.structural import StructuralLayer
from prompt_optimizer.types import CompressionContext
from tests.conftest import POLITE_PROMPT, MINIMAL_PROMPT


class TestStructuralLayer:
    def setup_method(self):
        self.layer = StructuralLayer()
        self.context = CompressionContext(
            agent_codes=["CEO", "COO", "CTO", "CFO", "CIO", "CMO"],
        )

    def test_strips_politeness(self):
        result = self.layer.compress(POLITE_PROMPT, self.context)
        assert "please" not in result.output_text.lower()
        assert "hey" not in result.output_text.lower()
        assert "could you" not in result.output_text.lower()

    def test_reduces_tokens(self):
        result = self.layer.compress(POLITE_PROMPT, self.context)
        assert result.output_tokens < result.input_tokens
        assert result.reduction_pct > 0.2  # At least 20% reduction

    def test_preserves_action_verb(self):
        result = self.layer.compress(POLITE_PROMPT, self.context)
        # "look at" maps to analyze; the output should contain an action
        output_lower = result.output_text.lower()
        assert any(a in output_lower for a in ["analyze", "review", "assess", "ANALYZE"])

    def test_preserves_agent_code(self):
        result = self.layer.compress(POLITE_PROMPT, self.context)
        assert "CFO" in result.output_text.upper()

    def test_preserves_time_reference(self):
        result = self.layer.compress(POLITE_PROMPT, self.context)
        assert "Q1" in result.output_text or "2026" in result.output_text

    def test_extracts_envelope(self):
        result = self.layer.compress(POLITE_PROMPT, self.context)
        xforms = str(result.transformations)
        assert "compiled to AST and rendered" in xforms or "compiled to AIL wire format" in xforms

    def test_minimal_prompt_no_error(self):
        result = self.layer.compress(MINIMAL_PROMPT, self.context)
        assert result.output_text  # Should produce something

    def test_risk_score_low(self):
        result = self.layer.compress(POLITE_PROMPT, self.context)
        assert result.risk_score <= 0.05

    def test_marked_reversible(self):
        result = self.layer.compress(POLITE_PROMPT, self.context)
        assert result.reversible is True

    def test_layer_level(self):
        assert self.layer.level == 1
        assert self.layer.name == "structural"

    def test_custom_filler_patterns(self):
        layer = StructuralLayer(extra_filler_patterns=[r"(?i)\byo\b"])
        result = layer.compress("Yo CFO, analyze the revenue data.", self.context)
        assert "yo" not in result.output_text.lower()

    def test_decompression(self):
        result = self.layer.compress(POLITE_PROMPT, self.context)
        decompressed = self.layer.decompress(result.output_text, self.context)
        assert decompressed  # Should produce readable text

    def test_priority_extraction(self):
        urgent = "ASAP, analyze the revenue numbers for Q1 2026."
        result = self.layer.compress(urgent, self.context)
        assert "urgent" in result.output_text.lower() or "!" in result.output_text

    def test_modifier_extraction(self):
        thorough = "Please thoroughly analyze the revenue data for Q1 2026."
        result = self.layer.compress(thorough, self.context)
        assert "thorough" in result.output_text.lower() or "~" in result.output_text

    def test_constraint_extraction(self):
        constrained = "Analyze revenue. Must complete within 2 hours. No more than 3 pages."
        result = self.layer.compress(constrained, self.context)
        # Constraints should be preserved in some form
        assert result.output_text  # At minimum, produces output

    def test_empty_input(self):
        result = self.layer.compress("", self.context)
        assert result.output_tokens <= result.input_tokens

    def test_already_compact_input(self):
        result = self.layer.compress("ANALYZE revenue Q1-2026", self.context)
        # Should not break already-compact input
        assert "revenue" in result.output_text.lower() or "revenue" in result.output_text
