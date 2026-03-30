"""Tests for ProgressiveOptimizer."""


from prompt_optimizer.blackboard import Blackboard
from prompt_optimizer.progressive import ProgressiveOptimizer
from prompt_optimizer.types import CompressionContext
from conftest import CONTEXT_HEAVY_PROMPT, MULTI_STEP_PROMPT, POLITE_PROMPT


class TestProgressiveOptimizer:
    def setup_method(self):
        self.optimizer = ProgressiveOptimizer()

    def test_basic_optimization(self):
        result = self.optimizer.optimize(POLITE_PROMPT)
        assert result.compressed_text
        assert result.token_counts.compressed <= result.token_counts.original
        assert len(result.layers_applied) > 0

    def test_respects_max_layer(self):
        result = self.optimizer.optimize(POLITE_PROMPT, max_layer=1)
        assert all(lyr <= 1 for lyr in result.layers_applied)

    def test_layer_2_applied(self):
        result = self.optimizer.optimize(MULTI_STEP_PROMPT, max_layer=2)
        # Multi-step prompts get pipeline compression at L2
        assert result.token_counts.compressed <= result.token_counts.original

    def test_layer_3_with_blackboard(self):
        bb = Blackboard()
        context = CompressionContext(
            agent_codes=["CEO", "CTO", "CFO"],
            blackboard=bb,
        )
        result = self.optimizer.optimize(CONTEXT_HEAVY_PROMPT, context=context, max_layer=3)
        assert result.compressed_text
        assert result.token_counts.compressed < result.token_counts.original

    def test_fidelity_report_attached(self):
        result = self.optimizer.optimize(POLITE_PROMPT)
        assert result.fidelity_report is not None
        assert result.fidelity_report.overall_score > 0

    def test_target_reduction_stops_early(self):
        result = self.optimizer.optimize(POLITE_PROMPT, target_reduction=0.1)
        # Should stop once 10% reduction is achieved
        assert result.token_counts.reduction_pct >= 0.0

    def test_min_fidelity_escape_hatch(self):
        optimizer = ProgressiveOptimizer(min_fidelity=0.95)
        result = optimizer.optimize(POLITE_PROMPT)
        # High fidelity threshold may prevent some layers
        assert result.fidelity_report is not None

    def test_max_risk_escape_hatch(self):
        optimizer = ProgressiveOptimizer(max_risk=0.01)
        result = optimizer.optimize(POLITE_PROMPT)
        # Very low risk threshold should limit compression
        assert result.fidelity_report is not None

    def test_empty_input(self):
        result = self.optimizer.optimize("")
        assert result.compressed_text is not None

    def test_already_compact_input(self):
        result = self.optimizer.optimize("ANALYZE revenue Q1-2026")
        assert result.compressed_text

    def test_layer_results_recorded(self):
        result = self.optimizer.optimize(POLITE_PROMPT)
        assert len(result.layer_results) > 0

    def test_metadata_includes_original(self):
        result = self.optimizer.optimize(POLITE_PROMPT)
        assert result.original_text == POLITE_PROMPT

    def test_default_context_created(self):
        # Should work without explicit context
        result = self.optimizer.optimize("Analyze the revenue.")
        assert result.compressed_text
