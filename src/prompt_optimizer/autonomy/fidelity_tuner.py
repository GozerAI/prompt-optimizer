"""Auto-adjust fidelity thresholds based on output quality feedback.

Tracks optimization outcomes and dynamically adjusts the minimum fidelity
threshold so the optimizer converges on the best tradeoff between compression
and quality for a given workload.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FidelityObservation:
    """A single feedback observation."""

    fidelity_threshold: float
    achieved_fidelity: float
    reduction_pct: float
    quality_score: float  # 0-1 user/system quality rating
    layer: int = 0


@dataclass
class TunerState:
    """Current state of the fidelity tuner."""

    current_threshold: float = 0.50
    observation_count: int = 0
    avg_quality: float = 0.0
    avg_reduction: float = 0.0
    trend: str = "stable"  # "improving", "degrading", "stable"


class FidelityTuner:
    """Dynamically tunes the minimum fidelity threshold based on feedback.

    Uses an exponential moving average of quality scores to adjust the
    fidelity floor up (more conservative) when quality drops and down
    (more aggressive compression) when quality is consistently high.
    """

    def __init__(
        self,
        *,
        initial_threshold: float = 0.50,
        min_bound: float = 0.30,
        max_bound: float = 0.95,
        learning_rate: float = 0.1,
        window_size: int = 50,
    ) -> None:
        self._threshold = max(min_bound, min(max_bound, initial_threshold))
        self._min_bound = min_bound
        self._max_bound = max_bound
        self._lr = learning_rate
        self._window_size = window_size
        self._observations: list[FidelityObservation] = []
        self._ema_quality: float = 0.75  # Start optimistic
        self._ema_reduction: float = 0.50

    @property
    def threshold(self) -> float:
        """Current recommended fidelity threshold."""
        return round(self._threshold, 4)

    def observe(self, obs: FidelityObservation) -> float:
        """Record an observation and return the updated threshold."""
        self._observations.append(obs)
        if len(self._observations) > self._window_size * 2:
            self._observations = self._observations[-self._window_size:]

        # Update exponential moving averages
        alpha = self._lr
        self._ema_quality = (1 - alpha) * self._ema_quality + alpha * obs.quality_score
        self._ema_reduction = (1 - alpha) * self._ema_reduction + alpha * obs.reduction_pct

        # Adjust threshold based on quality signal
        if self._ema_quality < 0.5:
            delta = self._lr * (0.7 - self._ema_quality)
            self._threshold = min(self._max_bound, self._threshold + delta)
        elif self._ema_quality > 0.8:
            delta = self._lr * (self._ema_quality - 0.8) * 0.5
            self._threshold = max(self._min_bound, self._threshold - delta)

        return self.threshold

    def get_state(self) -> TunerState:
        """Get current tuner state."""
        if len(self._observations) < 2:
            trend = "stable"
        else:
            recent_count = min(10, len(self._observations))
            recent = self._observations[-recent_count:]
            remaining = len(self._observations) - recent_count
            if remaining > 0:
                older = self._observations[:remaining]
                recent_avg = sum(o.quality_score for o in recent) / len(recent)
                older_avg = sum(o.quality_score for o in older) / len(older)
                diff = recent_avg - older_avg
                if diff > 0.05:
                    trend = "improving"
                elif diff < -0.05:
                    trend = "degrading"
                else:
                    trend = "stable"
            else:
                trend = "stable"

        return TunerState(
            current_threshold=self.threshold,
            observation_count=len(self._observations),
            avg_quality=round(self._ema_quality, 4),
            avg_reduction=round(self._ema_reduction, 4),
            trend=trend,
        )

    def reset(self) -> None:
        """Reset tuner to initial state."""
        self._observations.clear()
        self._ema_quality = 0.75
        self._ema_reduction = 0.50
