"""Prompt Optimizer — Three-layer prompt compression for inter-agent communication.

Layer 1 (Structural): Strips natural language into typed envelopes. ~65-70% token reduction.
Layer 2 (Semantic): Deduplicates context, pipeline shorthand. ~75-80% cumulative reduction.
Layer 3 (Context): Replaces shared context with blackboard pointers. ~90-95% cumulative reduction.
"""

from prompt_optimizer.blackboard import Blackboard
from prompt_optimizer.envelope import TypedEnvelope
from prompt_optimizer.fidelity import FidelityScorer
from prompt_optimizer.progressive import ProgressiveOptimizer
from prompt_optimizer.schema_registry import SchemaRegistry
from prompt_optimizer.types import (
    CompressedPrompt,
    CompressionContext,
    DriftFlag,
    FidelityReport,
    LayerFidelity,
    LayerResult,
    Recommendation,
    Severity,
    TokenCounts,
)
from prompt_optimizer.verifier import ReconstructionVerifier

__version__ = "0.1.0"

__all__ = [
    "Blackboard",
    "CompressedPrompt",
    "CompressionContext",
    "DriftFlag",
    "FidelityReport",
    "FidelityScorer",
    "LayerFidelity",
    "LayerResult",
    "ProgressiveOptimizer",
    "Recommendation",
    "ReconstructionVerifier",
    "SchemaRegistry",
    "Severity",
    "TokenCounts",
    "TypedEnvelope",
]


def optimize(
    text: str,
    max_layer: int = 2,
    min_fidelity: float = 0.50,
    context: CompressionContext | None = None,
) -> CompressedPrompt:
    """Quick-start function for prompt optimization.

    Args:
        text: The prompt to optimize.
        max_layer: Maximum layer (1=structural, 2=semantic, 3=context).
        min_fidelity: Minimum fidelity score before stopping.
        context: Optional compression context.

    Returns:
        CompressedPrompt with optimization results.
    """
    optimizer = ProgressiveOptimizer(min_fidelity=min_fidelity)
    return optimizer.optimize(text, context=context, max_layer=max_layer)
