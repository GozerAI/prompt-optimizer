"""Abstract base for compression layers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from prompt_optimizer.types import CompressionContext, LayerResult


class CompressionLayer(ABC):
    """Base class for a compression layer."""

    level: int
    name: str
    risk_range: tuple[float, float]  # (min_risk, max_risk)

    @abstractmethod
    def compress(self, text: str, context: CompressionContext) -> LayerResult:
        """Compress text, returning a LayerResult."""
        ...

    @abstractmethod
    def decompress(self, compressed: str, context: CompressionContext) -> str:
        """Reconstruct text from compressed form."""
        ...
