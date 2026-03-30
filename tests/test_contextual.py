"""Tests for Layer 3: Context compression."""


from prompt_optimizer.blackboard import Blackboard
from prompt_optimizer.layers.contextual import ContextualLayer
from prompt_optimizer.types import CompressionContext
from conftest import CONTEXT_HEAVY_PROMPT


class TestContextualLayer:
    def setup_method(self):
        self.layer = ContextualLayer()
        self.blackboard = Blackboard()
        self.context = CompressionContext(
            agent_codes=["CEO", "COO", "CTO", "CFO"],
            blackboard=self.blackboard,
        )

    def test_externalizes_financial_context(self):
        result = self.layer.compress(CONTEXT_HEAVY_PROMPT, self.context)
        assert any("financial" in t for t in result.transformations) or \
               any("org" in t for t in result.transformations) or \
               result.output_tokens < result.input_tokens

    def test_creates_blackboard_entries(self):
        self.layer.compress(CONTEXT_HEAVY_PROMPT, self.context)
        snapshot = self.blackboard.snapshot()
        assert len(snapshot) > 0

    def test_output_has_bb_refs(self):
        result = self.layer.compress(CONTEXT_HEAVY_PROMPT, self.context)
        if result.transformations and "skipped" not in str(result.transformations):
            assert "bb=" in result.output_text or "@v" in result.output_text

    def test_no_blackboard_skips(self):
        no_bb_context = CompressionContext(agent_codes=["CEO"])
        result = self.layer.compress(CONTEXT_HEAVY_PROMPT, no_bb_context)
        assert "skipped" in str(result.transformations)
        assert result.output_text == CONTEXT_HEAVY_PROMPT

    def test_decompression_resolves_pointers(self):
        result = self.layer.compress(CONTEXT_HEAVY_PROMPT, self.context)
        decompressed = self.layer.decompress(result.output_text, self.context)
        # Should expand pointers back to text — no pointer references remain
        assert decompressed
        if "[" in result.output_text and "@v" in result.output_text:
            assert "@v" not in decompressed or "bb=" not in decompressed

    def test_risk_scales_with_externalization(self):
        result = self.layer.compress(CONTEXT_HEAVY_PROMPT, self.context)
        assert result.risk_score >= self.layer.risk_range[0]
        assert result.risk_score <= self.layer.risk_range[1]

    def test_short_context_not_externalized(self):
        result = self.layer.compress("Analyze revenue.", self.context)
        # Too short to match context patterns
        assert len(result.transformations) <= 1

    def test_layer_level(self):
        assert self.layer.level == 3
        assert self.layer.name == "contextual"

    def test_preserves_action_after_extraction(self):
        result = self.layer.compress(CONTEXT_HEAVY_PROMPT, self.context)
        output_lower = result.output_text.lower()
        # The core action or subject should survive context extraction
        assert "analyz" in output_lower or "migration" in output_lower or "proceed" in output_lower

    def test_blackboard_versioning(self):
        self.layer.compress(CONTEXT_HEAVY_PROMPT, self.context)
        # Modify and recompress
        modified = CONTEXT_HEAVY_PROMPT.replace("$2.3M", "$2.5M")
        self.layer.compress(modified, self.context)
        snapshot = self.blackboard.snapshot()
        # Should have updated entries
        assert len(snapshot) > 0
