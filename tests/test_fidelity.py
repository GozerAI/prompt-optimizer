"""Tests for FidelityScorer."""

from prompt_optimizer.fidelity import FidelityScorer
from prompt_optimizer.types import (
    CompressedPrompt,
    LayerResult,
    Recommendation,
    TokenCounts,
)


class TestFidelityScorer:
    def setup_method(self):
        self.scorer = FidelityScorer()

    def test_identical_text_perfect_score(self):
        fidelity = self.scorer.score("Analyze revenue.", "Analyze revenue.", 1)
        assert fidelity.overall >= 0.9

    def test_score_decreases_with_info_loss(self):
        original = "CFO should analyze $2.3M Q1 2026 revenue with 12.4% growth."
        compressed = "Analyze revenue."
        fidelity = self.scorer.score(original, compressed, 1)
        assert fidelity.completeness < 1.0

    def test_score_all(self):
        compressed = CompressedPrompt(
            original_text="Analyze $2.3M revenue.",
            compressed_text="ANALYZE revenue",
            layers_applied=[1],
            token_counts=TokenCounts(original=5, compressed=2),
            layer_results=[
                LayerResult(
                    layer=1,
                    input_text="Analyze $2.3M revenue.",
                    output_text="ANALYZE revenue",
                    input_tokens=5,
                    output_tokens=2,
                    risk_score=0.02,
                ),
            ],
        )
        report = self.scorer.score_all(compressed)
        assert 0.0 <= report.overall_score <= 1.0
        assert len(report.per_layer) == 1

    def test_actionability_preserved(self):
        fidelity = self.scorer.score(
            "Analyze the revenue data for Q1 2026.",
            "ANALYZE revenue {period=Q1 2026}",
            1,
        )
        assert fidelity.actionability > 0.5

    def test_recommendation_safe(self):
        compressed = CompressedPrompt(
            original_text="Analyze revenue.",
            compressed_text="Analyze revenue.",
            layers_applied=[1],
            token_counts=TokenCounts(original=3, compressed=3),
            layer_results=[
                LayerResult(layer=1, input_text="Analyze revenue.",
                           output_text="Analyze revenue.", input_tokens=3,
                           output_tokens=3, risk_score=0.01),
            ],
        )
        report = self.scorer.score_all(compressed)
        assert report.recommendation == Recommendation.SAFE

    def test_empty_layer_results(self):
        compressed = CompressedPrompt(
            original_text="test",
            compressed_text="test",
            token_counts=TokenCounts(original=1, compressed=1),
        )
        report = self.scorer.score_all(compressed)
        assert report.overall_score == 1.0
