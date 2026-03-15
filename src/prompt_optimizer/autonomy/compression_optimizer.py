"""Auto-optimize compression ratio targets.

Learns the achievable compression ratio for different input types and
automatically adjusts target_reduction to maximize compression without
exceeding quality bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompressionProfile:
    """Statistics for a category of inputs."""

    category: str
    sample_count: int = 0
    avg_reduction: float = 0.0
    max_safe_reduction: float = 0.0
    min_quality_at_max: float = 1.0
    _reductions: list[float] = field(default_factory=list, repr=False)
    _qualities: list[float] = field(default_factory=list, repr=False)


@dataclass
class CompressionTarget:
    """Recommended compression target for an input."""

    target_reduction: float
    confidence: float
    category: str
    reasoning: str


class CompressionOptimizer:
    """Learns optimal compression targets from feedback."""

    def __init__(self, *, quality_floor: float = 0.65, safety_margin: float = 0.05) -> None:
        self._quality_floor = quality_floor
        self._safety_margin = safety_margin
        self._profiles: dict[str, CompressionProfile] = {}

    def categorize(self, token_count: int, has_structure: bool) -> str:
        if has_structure:
            return "structured"
        if token_count < 50:
            return "short"
        if token_count < 200:
            return "medium"
        return "large"

    def record(self, category: str, reduction_pct: float, quality_score: float) -> None:
        if category not in self._profiles:
            self._profiles[category] = CompressionProfile(category=category)
        prof = self._profiles[category]
        prof.sample_count += 1
        prof._reductions.append(reduction_pct)
        prof._qualities.append(quality_score)
        if len(prof._reductions) > 200:
            prof._reductions = prof._reductions[-200:]
            prof._qualities = prof._qualities[-200:]
        prof.avg_reduction = sum(prof._reductions) / len(prof._reductions)
        safe = [r for r, q in zip(prof._reductions, prof._qualities) if q >= self._quality_floor]
        if safe:
            prof.max_safe_reduction = max(safe)
            idx = prof._reductions.index(prof.max_safe_reduction)
            prof.min_quality_at_max = prof._qualities[idx]

    def recommend(self, category: str) -> CompressionTarget:
        prof = self._profiles.get(category)
        if prof is None or prof.sample_count < 3:
            defaults = {"short": 0.40, "medium": 0.55, "large": 0.65, "structured": 0.30}
            return CompressionTarget(
                target_reduction=defaults.get(category, 0.50), confidence=0.3,
                category=category, reasoning="Insufficient data, using default",
            )
        target = max(0.1, prof.max_safe_reduction - self._safety_margin)
        confidence = min(1.0, prof.sample_count / 20)
        return CompressionTarget(
            target_reduction=round(target, 3), confidence=round(confidence, 2),
            category=category,
            reasoning="Based on {} samples, max safe={:.1%}".format(prof.sample_count, prof.max_safe_reduction),
        )

    def get_profiles(self) -> dict[str, CompressionProfile]:
        return dict(self._profiles)

    def reset(self) -> None:
        self._profiles.clear()
