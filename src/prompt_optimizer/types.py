"""Core dataclasses for prompt optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Recommendation(str, Enum):
    SAFE = "safe"
    REVIEW = "review"
    UNSAFE = "unsafe"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class TokenCounts:
    original: int
    compressed: int

    @property
    def reduction_pct(self) -> float:
        if self.original == 0:
            return 0.0
        return 1.0 - (self.compressed / self.original)


@dataclass
class DriftFlag:
    """A specific semantic drift detected during verification."""

    layer: int
    category: str  # "missing_fact", "altered_value", "lost_constraint", "lost_nuance"
    description: str
    severity: Severity = Severity.WARNING
    original_fragment: str = ""
    compressed_fragment: str = ""


@dataclass
class LayerFidelity:
    layer: int
    completeness: float  # 0-1: are all key facts preserved?
    accuracy: float  # 0-1: are preserved facts unchanged?
    actionability: float  # 0-1: can recipient execute same action?

    @property
    def overall(self) -> float:
        return self.completeness * 0.4 + self.accuracy * 0.4 + self.actionability * 0.2


@dataclass
class LayerResult:
    layer: int
    input_text: str
    output_text: str
    input_tokens: int
    output_tokens: int
    risk_score: float
    transformations: list[str] = field(default_factory=list)
    reversible: bool = True

    @property
    def reduction_pct(self) -> float:
        if self.input_tokens == 0:
            return 0.0
        return 1.0 - (self.output_tokens / self.input_tokens)


@dataclass
class FidelityReport:
    overall_score: float
    per_layer: list[LayerFidelity] = field(default_factory=list)
    drift_flags: list[DriftFlag] = field(default_factory=list)
    recommendation: Recommendation = Recommendation.SAFE


@dataclass
class CompressedPrompt:
    original_text: str
    compressed_text: str
    envelope: Optional[Any] = None  # TypedEnvelope when available
    layers_applied: list[int] = field(default_factory=list)
    token_counts: TokenCounts = field(default_factory=lambda: TokenCounts(0, 0))
    fidelity_report: Optional[FidelityReport] = None
    blackboard_refs: list[str] = field(default_factory=list)
    layer_results: list[LayerResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionContext:
    """Context passed through compression layers."""

    history: list[str] = field(default_factory=list)
    agent_codes: list[str] = field(default_factory=list)
    blackboard: Optional[Any] = None  # Blackboard instance
    schema_registry: Optional[Any] = None  # SchemaRegistry instance
    conversation_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
