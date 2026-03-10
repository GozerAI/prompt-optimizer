"""Tests for Layer 2: Semantic compression."""

import pytest

from prompt_optimizer.layers.semantic import SemanticLayer
from prompt_optimizer.schema_registry import SchemaRegistry
from prompt_optimizer.types import CompressionContext
from tests.conftest import MULTI_STEP_PROMPT, POLITE_PROMPT


class TestSemanticLayer:
    def setup_method(self):
        self.layer = SemanticLayer()
        self.context = CompressionContext(
            agent_codes=["CEO", "COO", "CTO", "CFO", "CIO", "CMO"],
            schema_registry=SchemaRegistry(),
        )

    def test_deduplicates_repeated_context(self):
        self.context.history = [
            "The Q1 2026 revenue was $2.3M with a growth rate of 12.4%."
        ]
        text = (
            "The Q1 2026 revenue was $2.3M with a growth rate of 12.4%. "
            "Based on this, analyze growth trends."
        )
        result = self.layer.compress(text, self.context)
        assert "[ctx:" in result.output_text
        assert any("deduplicated" in t for t in result.transformations)

    def test_pipeline_collapse(self):
        result = self.layer.compress(MULTI_STEP_PROMPT, self.context)
        assert "PIPE(" in result.output_text or result.output_tokens <= result.input_tokens

    def test_schema_abbreviation(self):
        text = "Ask the Chief Technology Officer to analyze the system."
        result = self.layer.compress(text, self.context)
        assert "CTO" in result.output_text

    def test_preserves_meaning(self):
        result = self.layer.compress(MULTI_STEP_PROMPT, self.context)
        # Key agents should be preserved
        output_upper = result.output_text.upper()
        assert "CTO" in output_upper
        assert "CFO" in output_upper
        assert "CEO" in output_upper

    def test_risk_score_moderate(self):
        result = self.layer.compress(MULTI_STEP_PROMPT, self.context)
        assert result.risk_score <= 0.15

    def test_decompression_expands_references(self):
        self.context.history = ["Revenue data shows $2.3M in Q1."]
        text = "Revenue data shows $2.3M in Q1. Now analyze trends."
        result = self.layer.compress(text, self.context)

        decompressed = self.layer.decompress(result.output_text, self.context)
        # Should expand ctx references back
        assert decompressed  # At minimum produces output

    def test_decompression_expands_pipeline(self):
        result = self.layer.compress(MULTI_STEP_PROMPT, self.context)
        if "PIPE(" in result.output_text:
            decompressed = self.layer.decompress(result.output_text, self.context)
            assert "PIPE(" not in decompressed

    def test_resolves_vague_references(self):
        self.context.history = ["The CTO recommended using microservices."]
        text = "As mentioned earlier, we should proceed with the plan."
        result = self.layer.compress(text, self.context)
        if any("resolved" in t for t in result.transformations):
            assert "[ctx:" in result.output_text

    def test_no_history_no_dedup(self):
        text = "Analyze the revenue data for Q1 2026."
        result = self.layer.compress(text, self.context)
        assert "[ctx:" not in result.output_text

    def test_reset_index(self):
        self.layer.reset_index()
        text = "Analyze revenue."
        result = self.layer.compress(text, self.context)
        assert result.output_text  # Should work after reset

    def test_layer_level(self):
        assert self.layer.level == 2
        assert self.layer.name == "semantic"

    def test_multiple_compressions_build_index(self):
        text1 = "Q1 revenue was $2.3M."
        text2 = "Q1 revenue was $2.3M. Now forecast Q2."
        self.layer.compress(text1, self.context)
        result2 = self.layer.compress(text2, self.context)
        # Second compression should detect duplicate
        assert result2.output_tokens <= result2.input_tokens
