"""Self-tuning parameters based on feedback loops.

Orchestrates the autonomy sub-modules into a unified feedback loop that
tunes all optimization parameters together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prompt_optimizer.autonomy.auto_level import AutoLevelSelector, InputProfile, LevelRecommendation
from prompt_optimizer.autonomy.compression_optimizer import CompressionOptimizer
from prompt_optimizer.autonomy.fidelity_tuner import FidelityObservation, FidelityTuner


@dataclass
class TuningResult:
    """Parameters selected by the self-tuning engine."""
    max_layer: int = 2
    min_fidelity: float = 0.50
    target_reduction: float | None = None
    profile: InputProfile | None = None
    confidence: float = 0.0


@dataclass
class FeedbackRecord:
    """Feedback from a completed optimization."""
    original_tokens: int
    compressed_tokens: int
    quality_score: float
    fidelity_achieved: float
    layer_used: int
    category: str = ""


class SelfTuningEngine:
    """Unified self-tuning engine that orchestrates all autonomy components."""

    def __init__(self, *, conservative: bool = False, learning_rate: float = 0.1) -> None:
        self._level_selector = AutoLevelSelector(conservative=conservative)
        self._fidelity_tuner = FidelityTuner(learning_rate=learning_rate)
        self._compression_optimizer = CompressionOptimizer()
        self._feedback_count = 0

    def select(self, text: str) -> TuningResult:
        rec = self._level_selector.recommend(text)
        profile = self._level_selector.profile(text)
        min_fidelity = rec.min_fidelity
        if self._feedback_count >= 5:
            tuner_threshold = self._fidelity_tuner.threshold
            min_fidelity = 0.6 * tuner_threshold + 0.4 * rec.min_fidelity
        target_reduction = rec.target_reduction
        category = self._compression_optimizer.categorize(
            profile.token_count, profile.has_structured_syntax
        )
        if self._feedback_count >= 5:
            comp_target = self._compression_optimizer.recommend(category)
            if comp_target.confidence > 0.5:
                target_reduction = comp_target.target_reduction
        return TuningResult(
            max_layer=rec.max_layer, min_fidelity=round(min_fidelity, 3),
            target_reduction=target_reduction, profile=profile, confidence=rec.confidence,
        )

    def feedback(self, record: FeedbackRecord) -> None:
        self._feedback_count += 1
        profile = InputProfile(token_count=record.original_tokens)
        self._level_selector.record_outcome(profile, record.quality_score)
        reduction = 1.0 - (record.compressed_tokens / max(1, record.original_tokens))
        obs = FidelityObservation(
            fidelity_threshold=0.0, achieved_fidelity=record.fidelity_achieved,
            reduction_pct=reduction, quality_score=record.quality_score, layer=record.layer_used,
        )
        self._fidelity_tuner.observe(obs)
        category = record.category or self._compression_optimizer.categorize(record.original_tokens, False)
        self._compression_optimizer.record(category, reduction, record.quality_score)

    @property
    def feedback_count(self) -> int:
        return self._feedback_count

    @property
    def level_selector(self) -> AutoLevelSelector:
        return self._level_selector

    @property
    def fidelity_tuner(self) -> FidelityTuner:
        return self._fidelity_tuner

    @property
    def compression_optimizer(self) -> CompressionOptimizer:
        return self._compression_optimizer

    def reset(self) -> None:
        self._level_selector = AutoLevelSelector()
        self._fidelity_tuner.reset()
        self._compression_optimizer.reset()
        self._feedback_count = 0
