"""Tests for LLM-as-Judge fidelity scoring."""

from unittest.mock import AsyncMock

import pytest

from prompt_optimizer.fidelity import FidelityScorer
from prompt_optimizer.fidelity_llm import (
    CalibrationResult,
    CompositeFidelityScorer,
    LLMFidelityScorer,
    LLMFidelityVerdict,
)


class TestLLMFidelityVerdict:
    def test_overall_score(self):
        v = LLMFidelityVerdict(
            intent_preservation=0.8,
            context_sufficiency=0.6,
            actionability=1.0,
        )
        # 0.8*0.4 + 0.6*0.4 + 1.0*0.2 = 0.32 + 0.24 + 0.20 = 0.76
        assert abs(v.overall - 0.76) < 0.001

    def test_zero_scores(self):
        v = LLMFidelityVerdict(
            intent_preservation=0.0,
            context_sufficiency=0.0,
            actionability=0.0,
        )
        assert v.overall == 0.0

    def test_perfect_scores(self):
        v = LLMFidelityVerdict(
            intent_preservation=1.0,
            context_sufficiency=1.0,
            actionability=1.0,
        )
        assert v.overall == 1.0

    def test_error_field(self):
        v = LLMFidelityVerdict(
            intent_preservation=0.0,
            context_sufficiency=0.0,
            actionability=0.0,
            error="connection failed",
        )
        assert v.error == "connection failed"


class TestLLMFidelityScorerParsing:
    def setup_method(self):
        self.scorer = LLMFidelityScorer()

    def test_parse_valid_json(self):
        response = '{"intent_preservation": 5, "context_sufficiency": 4, "actionability": 3, "reasoning": "good"}'
        verdict = self.scorer._parse_verdict(response)
        assert verdict.intent_preservation == 1.0  # (5-1)/4 = 1.0
        assert verdict.context_sufficiency == 0.75  # (4-1)/4 = 0.75
        assert verdict.actionability == 0.5  # (3-1)/4 = 0.5
        assert verdict.reasoning == "good"

    def test_parse_json_with_surrounding_text(self):
        response = 'Here is my evaluation:\n{"intent_preservation": 4, "context_sufficiency": 4, "actionability": 4, "reasoning": "ok"}\nDone.'
        verdict = self.scorer._parse_verdict(response)
        assert verdict.intent_preservation == 0.75
        assert verdict.error is None

    def test_parse_no_json(self):
        verdict = self.scorer._parse_verdict("This has no JSON.")
        assert verdict.error is not None
        assert verdict.intent_preservation == 0.0

    def test_parse_invalid_json(self):
        verdict = self.scorer._parse_verdict("{broken json!!!}")
        assert verdict.error is not None

    def test_parse_missing_fields(self):
        response = '{"intent_preservation": 3}'
        verdict = self.scorer._parse_verdict(response)
        assert verdict.intent_preservation == 0.5
        assert verdict.context_sufficiency == 0.0  # missing → 0

    def test_parse_scores_clamped(self):
        response = '{"intent_preservation": 10, "context_sufficiency": -1, "actionability": 5, "reasoning": ""}'
        verdict = self.scorer._parse_verdict(response)
        assert verdict.intent_preservation == 1.0  # clamped
        assert verdict.context_sufficiency == 0.0  # clamped


class TestLLMFidelityScorerAsync:
    @pytest.mark.asyncio
    async def test_score_handles_call_error(self):
        scorer = LLMFidelityScorer()
        scorer._call_llm = AsyncMock(side_effect=RuntimeError("connection refused"))
        verdict = await scorer.score("original", "compressed")
        assert verdict.error is not None

    @pytest.mark.asyncio
    async def test_score_success(self):
        scorer = LLMFidelityScorer()
        scorer._call_llm = AsyncMock(return_value='{"intent_preservation": 5, "context_sufficiency": 4, "actionability": 5, "reasoning": "good match"}')
        verdict = await scorer.score("Analyze revenue", "ANALYZE revenue")
        assert verdict.error is None
        assert verdict.intent_preservation == 1.0
        assert verdict.reasoning == "good match"

    @pytest.mark.asyncio
    async def test_batch_calibrate(self):
        scorer = LLMFidelityScorer()
        scorer._call_llm = AsyncMock(return_value='{"intent_preservation": 4, "context_sufficiency": 4, "actionability": 4, "reasoning": "ok"}')

        rule_scorer = FidelityScorer()
        samples = [
            ("Analyze the Q1 2026 revenue data.", "ANALYZE revenue {period=Q1 2026}"),
            ("Generate a financial report.", "GENERATE report"),
        ]

        result = await scorer.batch_calibrate(samples, rule_scorer)
        assert isinstance(result, CalibrationResult)
        assert result.samples == 2
        assert 0.0 <= result.mean_agreement <= 1.0


class TestCompositeFidelityScorer:
    def test_rule_only_by_default(self):
        composite = CompositeFidelityScorer()
        fidelity = composite.score("Analyze revenue.", "ANALYZE revenue", 1)
        assert 0.0 <= fidelity.overall <= 1.0

    def test_score_all(self):
        from prompt_optimizer.types import CompressedPrompt, LayerResult, TokenCounts

        composite = CompositeFidelityScorer()
        compressed = CompressedPrompt(
            original_text="Analyze revenue.",
            compressed_text="ANALYZE revenue",
            layers_applied=[1],
            token_counts=TokenCounts(original=3, compressed=2),
            layer_results=[
                LayerResult(layer=1, input_text="Analyze revenue.",
                           output_text="ANALYZE revenue", input_tokens=3,
                           output_tokens=2, risk_score=0.01),
            ],
        )
        report = composite.score_all(compressed)
        assert report.overall_score > 0

    @pytest.mark.asyncio
    async def test_score_with_llm_disabled(self):
        composite = CompositeFidelityScorer(use_llm=False)
        fidelity = await composite.score_with_llm("original", "compressed", 1)
        assert 0.0 <= fidelity.overall <= 1.0

    @pytest.mark.asyncio
    async def test_score_with_llm_enabled(self):
        mock_llm = LLMFidelityScorer()
        mock_llm._call_llm = AsyncMock(
            return_value='{"intent_preservation": 5, "context_sufficiency": 5, "actionability": 5, "reasoning": "perfect"}'
        )
        composite = CompositeFidelityScorer(
            llm_scorer=mock_llm, use_llm=True, llm_weight=0.3
        )
        fidelity = await composite.score_with_llm("Analyze revenue.", "ANALYZE revenue", 1)
        assert fidelity.overall > 0

    @pytest.mark.asyncio
    async def test_score_with_llm_error_falls_back(self):
        mock_llm = LLMFidelityScorer()
        mock_llm._call_llm = AsyncMock(side_effect=RuntimeError("boom"))
        composite = CompositeFidelityScorer(
            llm_scorer=mock_llm, use_llm=True
        )
        fidelity = await composite.score_with_llm("Analyze revenue.", "ANALYZE revenue", 1)
        # Should fall back to rule-based
        assert 0.0 <= fidelity.overall <= 1.0
