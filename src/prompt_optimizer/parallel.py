"""Parallel layer application for independent optimization passes.

When layers are independent (e.g., multiple structural sub-passes),
they can be applied in parallel using ThreadPoolExecutor. The results
are then merged, picking the best compression for each section.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import Any

from prompt_optimizer.fidelity import FidelityScorer
from prompt_optimizer.layers.base import CompressionLayer
from prompt_optimizer.tokenizer import count_tokens
from prompt_optimizer.types import CompressionContext, LayerResult


@dataclass
class ParallelResult:
    """Result of parallel layer application."""

    best_output: str
    best_layer: int
    all_results: list[LayerResult]
    parallelism_used: int


def _apply_layer(
    layer: CompressionLayer,
    text: str,
    context: CompressionContext,
) -> LayerResult:
    """Apply a single layer (thread-safe)."""
    return layer.compress(text, context)


class ParallelLayerApplicator:
    """Applies independent layers in parallel and selects the best result.

    Independent layers are those at the same level (e.g., multiple L1 strategies)
    or layers that don't depend on each other's output.
    """

    def __init__(
        self,
        scorer: FidelityScorer | None = None,
        max_workers: int = 4,
    ) -> None:
        self._scorer = scorer or FidelityScorer()
        self._max_workers = max_workers

    def apply_parallel(
        self,
        text: str,
        layers: list[CompressionLayer],
        context: CompressionContext | None = None,
    ) -> ParallelResult:
        """Apply all given layers in parallel and return the best result.

        Each layer is applied independently to the same input text.
        The result with the best fidelity-to-compression ratio is selected.
        """
        if context is None:
            context = CompressionContext()

        if len(layers) <= 1:
            # No parallelism needed
            if layers:
                result = layers[0].compress(text, context)
                return ParallelResult(
                    best_output=result.output_text,
                    best_layer=result.layer,
                    all_results=[result],
                    parallelism_used=1,
                )
            return ParallelResult(
                best_output=text,
                best_layer=0,
                all_results=[],
                parallelism_used=0,
            )

        results: list[LayerResult] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(_apply_layer, layer, text, context): layer
                for layer in layers
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception:
                    # Layer failed — skip it
                    pass

        if not results:
            return ParallelResult(
                best_output=text,
                best_layer=0,
                all_results=[],
                parallelism_used=len(layers),
            )

        # Score each result and pick best
        best: LayerResult | None = None
        best_score = -1.0

        for result in results:
            if result.output_text == result.input_text:
                # Layer did nothing
                score = 0.0
            else:
                fidelity = self._scorer.score(text, result.output_text, result.layer)
                reduction = result.reduction_pct
                # Combined score: fidelity * compression (favor high fidelity + good compression)
                score = fidelity.overall * (0.5 + reduction * 0.5)

            if score > best_score:
                best_score = score
                best = result

        if best is None or best.output_text == best.input_text:
            return ParallelResult(
                best_output=text,
                best_layer=0,
                all_results=results,
                parallelism_used=len(layers),
            )

        return ParallelResult(
            best_output=best.output_text,
            best_layer=best.layer,
            all_results=results,
            parallelism_used=len(layers),
        )

    def apply_layer_groups(
        self,
        text: str,
        layer_groups: list[list[CompressionLayer]],
        context: CompressionContext | None = None,
    ) -> list[ParallelResult]:
        """Apply groups of layers sequentially, with parallelism within each group.

        Each group's output feeds into the next group.
        """
        if context is None:
            context = CompressionContext()

        results: list[ParallelResult] = []
        current_text = text

        for group in layer_groups:
            result = self.apply_parallel(current_text, group, context)
            results.append(result)
            current_text = result.best_output

        return results
