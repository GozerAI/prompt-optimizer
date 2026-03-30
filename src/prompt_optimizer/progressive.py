"""Progressive optimizer — applies compression layers incrementally with escape hatches."""

from __future__ import annotations

from prompt_optimizer.fidelity import FidelityScorer
from prompt_optimizer.layers.base import CompressionLayer
from prompt_optimizer.tokenizer import count_tokens
from prompt_optimizer.types import (
    CompressedPrompt,
    CompressionContext,
    LayerResult,
    TokenCounts,
)
from prompt_optimizer.verifier import ReconstructionVerifier


class ProgressiveOptimizer:
    """Applies compression layers incrementally, stopping when fidelity drops too low.

    Each layer is applied and scored. If the fidelity score drops below
    min_fidelity or risk exceeds max_risk, the optimizer stops and returns
    the previous layer's result.
    """

    def __init__(
        self,
        layers: list[CompressionLayer] | None = None,
        verifier: ReconstructionVerifier | None = None,
        scorer: FidelityScorer | None = None,
        min_fidelity: float = 0.50,
        max_risk: float = 0.25,
    ) -> None:
        from prompt_optimizer.layers.contextual import ContextualLayer
        from prompt_optimizer.layers.semantic import SemanticLayer
        from prompt_optimizer.layers.structural import StructuralLayer

        self.layers = layers or [StructuralLayer(), SemanticLayer(), ContextualLayer()]
        self.verifier = verifier or ReconstructionVerifier()
        self.scorer = scorer or FidelityScorer()
        self.min_fidelity = min_fidelity
        self.max_risk = max_risk

    def optimize(
        self,
        text: str,
        context: CompressionContext | None = None,
        target_reduction: float | None = None,
        max_layer: int = 3,
    ) -> CompressedPrompt:
        """Apply layers incrementally up to max_layer.

        Args:
            text: The prompt text to optimize.
            context: Compression context with history, blackboard, etc.
            target_reduction: Stop when this token reduction % is reached (0.0-1.0).
            max_layer: Maximum layer to apply (1, 2, or 3).

        Returns:
            CompressedPrompt with the best compression within safety bounds.
        """
        if context is None:
            context = CompressionContext()

        original_tokens = count_tokens(text)
        current_text = text
        layer_results: list[LayerResult] = []
        layers_applied: list[int] = []
        best_text = text
        best_layers: list[int] = []
        best_results: list[LayerResult] = []
        applied_layers: list[CompressionLayer] = []

        for layer in self.layers:
            if layer.level > max_layer:
                break

            # Apply layer
            result = layer.compress(current_text, context)
            layer_results.append(result)

            # Check if layer did anything
            if result.output_text == result.input_text:
                continue

            # Score fidelity
            fidelity = self.scorer.score(text, result.output_text, layer.level)

            # Check escape hatches
            if fidelity.overall < self.min_fidelity:
                # Fidelity too low — stop, use previous best
                break

            if result.risk_score > self.max_risk:
                # Risk too high — stop
                break

            # Layer passed checks — accept it
            current_text = result.output_text
            layers_applied.append(layer.level)
            applied_layers.append(layer)
            best_text = current_text
            best_layers = list(layers_applied)
            best_results = list(layer_results)

            # Check if target reduction reached
            if target_reduction is not None:
                current_tokens = count_tokens(current_text)
                current_reduction = 1.0 - (current_tokens / max(original_tokens, 1))
                if current_reduction >= target_reduction:
                    break

        # Build final result
        compressed_tokens = count_tokens(best_text)

        # Run verification on final result
        compressed = CompressedPrompt(
            original_text=text,
            compressed_text=best_text,
            layers_applied=best_layers,
            token_counts=TokenCounts(original=original_tokens, compressed=compressed_tokens),
            layer_results=best_results,
        )

        # Full verification
        fidelity_report = self.verifier.verify(
            original=text,
            compressed=compressed,
            layers=applied_layers,
            context=context,
        )
        compressed.fidelity_report = fidelity_report

        return compressed
