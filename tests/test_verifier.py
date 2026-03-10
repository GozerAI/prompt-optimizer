"""Tests for ReconstructionVerifier."""

from prompt_optimizer.types import (
    CompressedPrompt,
    CompressionContext,
    LayerResult,
    Recommendation,
    TokenCounts,
)
from prompt_optimizer.verifier import ReconstructionVerifier


class TestReconstructionVerifier:
    def setup_method(self):
        self.verifier = ReconstructionVerifier()

    def test_identical_text_high_fidelity(self):
        compressed = CompressedPrompt(
            original_text="Analyze the revenue.",
            compressed_text="Analyze the revenue.",
            layers_applied=[],
            token_counts=TokenCounts(original=5, compressed=5),
            layer_results=[],
        )
        report = self.verifier.verify("Analyze the revenue.", compressed)
        assert report.overall_score >= 0.9
        assert report.recommendation == Recommendation.SAFE

    def test_detects_missing_agent_code(self):
        compressed = CompressedPrompt(
            original_text="CFO should analyze revenue.",
            compressed_text="Analyze revenue.",
            layers_applied=[1],
            token_counts=TokenCounts(original=6, compressed=3),
            layer_results=[
                LayerResult(layer=1, input_text="CFO should analyze revenue.",
                           output_text="Analyze revenue.", input_tokens=6,
                           output_tokens=3, risk_score=0.02),
            ],
        )
        report = self.verifier.verify("CFO should analyze revenue.", compressed)
        assert any("CFO" in str(d.description) for d in report.drift_flags)

    def test_detects_missing_numbers(self):
        original = "Revenue was $2.3M with 12.4% growth."
        compressed = CompressedPrompt(
            original_text=original,
            compressed_text="Revenue had growth.",
            layers_applied=[1],
            token_counts=TokenCounts(original=10, compressed=4),
            layer_results=[
                LayerResult(layer=1, input_text=original,
                           output_text="Revenue had growth.", input_tokens=10,
                           output_tokens=4, risk_score=0.05),
            ],
        )
        report = self.verifier.verify(original, compressed)
        assert any("number" in d.description.lower() or "2.3" in d.description
                   for d in report.drift_flags)

    def test_safe_recommendation_for_good_compression(self):
        original = "Please analyze the Q1 2026 revenue data."
        compressed_text = "ANALYZE revenue {period=Q1 2026}"
        compressed = CompressedPrompt(
            original_text=original,
            compressed_text=compressed_text,
            layers_applied=[1],
            token_counts=TokenCounts(original=10, compressed=5),
            layer_results=[
                LayerResult(layer=1, input_text=original,
                           output_text=compressed_text, input_tokens=10,
                           output_tokens=5, risk_score=0.02),
            ],
        )
        report = self.verifier.verify(original, compressed)
        # Should preserve key elements
        assert report.recommendation in (Recommendation.SAFE, Recommendation.REVIEW)

    def test_unsafe_recommendation_for_bad_compression(self):
        original = "CFO must analyze $2.3M Q1 2026 revenue with 12.4% growth and 20% margins."
        compressed = CompressedPrompt(
            original_text=original,
            compressed_text="Do stuff.",
            layers_applied=[1, 2, 3],
            token_counts=TokenCounts(original=20, compressed=2),
            layer_results=[
                LayerResult(layer=1, input_text=original,
                           output_text="Do stuff.", input_tokens=20,
                           output_tokens=2, risk_score=0.2),
            ],
        )
        report = self.verifier.verify(original, compressed)
        assert report.recommendation == Recommendation.UNSAFE

    def test_actionability_scoring(self):
        original = "Analyze the revenue data."
        compressed = CompressedPrompt(
            original_text=original,
            compressed_text="ANALYZE revenue",
            layers_applied=[1],
            token_counts=TokenCounts(original=6, compressed=2),
            layer_results=[
                LayerResult(layer=1, input_text=original,
                           output_text="ANALYZE revenue", input_tokens=6,
                           output_tokens=2, risk_score=0.01),
            ],
        )
        report = self.verifier.verify(original, compressed)
        if report.per_layer:
            assert report.per_layer[0].actionability > 0.5
