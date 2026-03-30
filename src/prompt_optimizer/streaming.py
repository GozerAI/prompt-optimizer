"""Streaming optimization for large prompts.

Splits large prompts into manageable sections and optimizes them incrementally,
yielding partial results as each section completes. Also supports incremental
optimization (only re-optimizing changed sections).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Generator

from prompt_optimizer.layers.base import CompressionLayer
from prompt_optimizer.tokenizer import count_tokens
from prompt_optimizer.types import (
    CompressedPrompt,
    CompressionContext,
    LayerResult,
    TokenCounts,
)


@dataclass
class SectionResult:
    """Result of optimizing a single section."""

    section_index: int
    original_text: str
    compressed_text: str
    content_hash: str
    tokens_saved: int
    from_cache: bool = False


@dataclass
class StreamingProgress:
    """Progress update during streaming optimization."""

    sections_total: int
    sections_done: int
    current_section: int
    tokens_original: int
    tokens_compressed: int
    section_result: SectionResult | None = None

    @property
    def progress_pct(self) -> float:
        if self.sections_total == 0:
            return 1.0
        return self.sections_done / self.sections_total


def _section_hash(text: str) -> str:
    """Hash a section for change detection."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def split_into_sections(
    text: str, max_section_tokens: int = 500
) -> list[str]:
    """Split text into logical sections for streaming optimization.

    Splits on paragraph boundaries, then sentence boundaries if paragraphs
    are too large. Preserves logical grouping.
    """
    if count_tokens(text) <= max_section_tokens:
        return [text]

    # Try paragraph split first
    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) > 1:
        sections: list[str] = []
        current = ""
        for para in paragraphs:
            candidate = (current + "\n\n" + para).strip() if current else para.strip()
            if count_tokens(candidate) > max_section_tokens and current:
                sections.append(current.strip())
                current = para.strip()
            else:
                current = candidate
        if current.strip():
            sections.append(current.strip())
        if len(sections) > 1:
            return sections

    # Fall back to sentence split
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sections = []
    current = ""
    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if count_tokens(candidate) > max_section_tokens and current:
            sections.append(current.strip())
            current = sentence
        else:
            current = candidate
    if current.strip():
        sections.append(current.strip())

    return sections if sections else [text]


class StreamingOptimizer:
    """Streaming optimizer that processes sections incrementally.

    Yields StreamingProgress updates as each section completes.
    Supports caching of section results for incremental re-optimization.
    """

    def __init__(
        self,
        layers: list[CompressionLayer] | None = None,
        max_section_tokens: int = 500,
    ) -> None:
        from prompt_optimizer.layers.contextual import ContextualLayer
        from prompt_optimizer.layers.semantic import SemanticLayer
        from prompt_optimizer.layers.structural import StructuralLayer

        self._layers = layers or [StructuralLayer(), SemanticLayer(), ContextualLayer()]
        self._max_section_tokens = max_section_tokens
        # Section cache: hash -> compressed text
        self._section_cache: dict[str, str] = {}

    def optimize_streaming(
        self,
        text: str,
        context: CompressionContext | None = None,
        max_layer: int = 2,
    ) -> Generator[StreamingProgress, None, CompressedPrompt]:
        """Optimize text in streaming fashion, yielding progress updates.

        Usage:
            gen = optimizer.optimize_streaming(text)
            try:
                while True:
                    progress = next(gen)
                    print(f"{progress.progress_pct:.0%} done")
            except StopIteration as e:
                result = e.value  # CompressedPrompt
        """
        if context is None:
            context = CompressionContext()

        sections = split_into_sections(text, self._max_section_tokens)
        total_original_tokens = count_tokens(text)
        section_results: list[SectionResult] = []
        compressed_sections: list[str] = []
        total_compressed_tokens = 0

        for i, section in enumerate(sections):
            section_h = _section_hash(section)

            # Check section cache
            cached = self._section_cache.get(section_h)
            if cached is not None:
                compressed_text = cached
                from_cache = True
            else:
                # Apply layers to this section
                compressed_text = section
                for layer in self._layers:
                    if layer.level > max_layer:
                        break
                    result = layer.compress(compressed_text, context)
                    if result.output_text != result.input_text:
                        compressed_text = result.output_text

                self._section_cache[section_h] = compressed_text
                from_cache = False

            section_tokens = count_tokens(compressed_text)
            total_compressed_tokens += section_tokens
            compressed_sections.append(compressed_text)

            sr = SectionResult(
                section_index=i,
                original_text=section,
                compressed_text=compressed_text,
                content_hash=section_h,
                tokens_saved=count_tokens(section) - section_tokens,
                from_cache=from_cache,
            )
            section_results.append(sr)

            yield StreamingProgress(
                sections_total=len(sections),
                sections_done=i + 1,
                current_section=i,
                tokens_original=total_original_tokens,
                tokens_compressed=total_compressed_tokens,
                section_result=sr,
            )

        final_text = "\n\n".join(compressed_sections)
        final_tokens = count_tokens(final_text)

        return CompressedPrompt(
            original_text=text,
            compressed_text=final_text,
            layers_applied=list(range(1, max_layer + 1)),
            token_counts=TokenCounts(original=total_original_tokens, compressed=final_tokens),
            metadata={
                "streaming": True,
                "sections": len(sections),
                "cache_hits": sum(1 for sr in section_results if sr.from_cache),
            },
        )

    def optimize_incremental(
        self,
        text: str,
        context: CompressionContext | None = None,
        max_layer: int = 2,
    ) -> CompressedPrompt:
        """Optimize only changed sections (uses section cache).

        Same as optimize_streaming but consumes the generator internally.
        """
        gen = self.optimize_streaming(text, context, max_layer)
        try:
            while True:
                next(gen)
        except StopIteration as e:
            return e.value

    def invalidate_cache(self) -> None:
        """Clear the section cache."""
        self._section_cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._section_cache)
