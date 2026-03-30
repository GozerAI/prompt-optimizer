"""Layer 2: Semantic compression — deduplicate context, resolve references, pipeline shorthand."""

from __future__ import annotations

import hashlib
import re

from prompt_optimizer.layers.base import CompressionLayer
from prompt_optimizer.tokenizer import count_tokens
from prompt_optimizer.types import CompressionContext, LayerResult


class SemanticIndex:
    """Tracks seen context blocks for deduplication."""

    def __init__(self) -> None:
        self._seen: dict[str, str] = {}  # content_hash -> reference_id
        self._store: dict[str, str] = {}  # reference_id -> original text
        self._counter = 0

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        return re.sub(r"\s+", " ", text.lower().strip())

    def _hash(self, text: str) -> str:
        return hashlib.sha256(self._normalize(text).encode()).hexdigest()[:16]

    def check_duplicate(self, text: str) -> str | None:
        """If text was seen before, return its reference id."""
        h = self._hash(text)
        return self._seen.get(h)

    def register(self, text: str) -> str:
        """Register text, return reference id."""
        h = self._hash(text)
        if h in self._seen:
            return self._seen[h]

        self._counter += 1
        ref_id = f"ctx:{self._counter}"
        self._seen[h] = ref_id
        self._store[ref_id] = text
        return ref_id

    def resolve(self, ref_id: str) -> str | None:
        """Resolve a reference id back to text."""
        return self._store.get(ref_id)


# Patterns for pipeline detection
SEQUENCE_PATTERNS = [
    r"(?i)\b(?:first|step\s*1)[,:]?\s*(.+?)(?:\.\s*|\n)",
    r"(?i)\b(?:then|next|step\s*2|after\s+that)[,:]?\s*(.+?)(?:\.\s*|\n)",
    r"(?i)\b(?:finally|lastly|step\s*3|last)[,:]?\s*(.+?)(?:\.\s*|\n|$)",
]

PIPE_KEYWORDS = [
    r"(?i)\bfirst\b.+?\bthen\b",
    r"(?i)\bstep\s*\d\b",
    r"(?i)\bafter\s+(?:that|this|which)\b",
    r"(?i)\b(?:and\s+)?finally\b",
]


class SemanticLayer(CompressionLayer):
    """Layer 2: Semantic compression.

    - Deduplicates repeated context across conversation history
    - Resolves forward/backward references
    - Collapses sequential instructions into pipeline notation
    """

    level = 2
    name = "semantic"
    risk_range = (0.05, 0.12)

    def __init__(self) -> None:
        self._index = SemanticIndex()

    def compress(self, text: str, context: CompressionContext) -> LayerResult:
        input_tokens = count_tokens(text)
        transformations: list[str] = []
        result_text = text

        # Step 1: Deduplicate context from history
        result_text, dedup_transforms = self._deduplicate_context(result_text, context)
        transformations.extend(dedup_transforms)

        # Step 2: Resolve references ("as mentioned earlier", "the previous analysis")
        result_text, ref_transforms = self._resolve_references(result_text, context)
        transformations.extend(ref_transforms)

        # Step 3: Collapse sequential instructions to pipeline
        result_text, pipe_transforms = self._collapse_to_pipeline(result_text, context)
        transformations.extend(pipe_transforms)

        # Step 4: Apply schema abbreviations if registry available
        if context.schema_registry:
            before = result_text
            result_text = context.schema_registry.abbreviate(result_text)
            if result_text != before:
                transformations.append("applied schema abbreviations")

        output_tokens = count_tokens(result_text)

        risk = self.risk_range[0]
        if transformations:
            # More transformations = more risk
            risk = min(self.risk_range[1], self.risk_range[0] + len(transformations) * 0.015)

        return LayerResult(
            layer=2,
            input_text=text,
            output_text=result_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            risk_score=risk,
            transformations=transformations,
            reversible=True,
        )

    def decompress(self, compressed: str, context: CompressionContext) -> str:
        """Expand compressed text back using semantic index."""
        result = compressed

        # Expand context references
        for match in re.finditer(r"\[ctx:(\d+)\]", result):
            ref_id = f"ctx:{match.group(1)}"
            resolved = self._index.resolve(ref_id)
            if resolved:
                result = result.replace(match.group(0), resolved)

        # Expand pipeline notation
        pipe_match = re.search(r"PIPE\((.+?)\)", result)
        if pipe_match:
            steps = [s.strip() for s in pipe_match.group(1).split("→")]
            ordinals = ["First", "Then", "Finally"]
            expanded_steps = []
            for i, step in enumerate(steps):
                prefix = ordinals[i] if i < len(ordinals) else f"Step {i + 1}"
                expanded_steps.append(f"{prefix}, {step}.")
            result = result.replace(pipe_match.group(0), " ".join(expanded_steps))

        # Expand schema abbreviations
        if context.schema_registry:
            result = context.schema_registry.expand(result)

        return result

    def _deduplicate_context(
        self, text: str, context: CompressionContext
    ) -> tuple[str, list[str]]:
        """Replace repeated context blocks with references."""
        transforms: list[str] = []

        # Register history entries
        for entry in context.history:
            self._index.register(entry)

        # Check if any sentence in the text duplicates history
        sentences = re.split(r"(?<=[.!?])\s+", text)
        new_sentences: list[str] = []

        for sentence in sentences:
            if not sentence.strip():
                continue

            ref = self._index.check_duplicate(sentence)
            if ref:
                new_sentences.append(f"[{ref}]")
                transforms.append(f"deduplicated: '{sentence[:50]}...' -> [{ref}]")
            else:
                self._index.register(sentence)
                new_sentences.append(sentence)

        return " ".join(new_sentences), transforms

    def _resolve_references(
        self, text: str, context: CompressionContext
    ) -> tuple[str, list[str]]:
        """Resolve vague references to specific ones."""
        transforms: list[str] = []

        # Replace "as mentioned earlier/before" with context pointer if available
        ref_patterns = [
            r"(?i)as\s+(?:I\s+)?mentioned\s+(?:earlier|before|previously)",
            r"(?i)the\s+(?:previous|earlier|prior)\s+(?:analysis|report|assessment|discussion)",
            r"(?i)(?:based on|per|according to)\s+(?:our|the)\s+(?:earlier|previous)\s+(?:conversation|discussion|analysis)",
        ]

        for pattern in ref_patterns:
            match = re.search(pattern, text)
            if match and context.history:
                # Reference the most recent history entry
                ref = self._index.register(context.history[-1])
                text = text[:match.start()] + f"[{ref}]" + text[match.end():]
                transforms.append(f"resolved reference: '{match.group()}'")

        return text, transforms

    def _collapse_to_pipeline(
        self, text: str, context: CompressionContext
    ) -> tuple[str, list[str]]:
        """Collapse sequential multi-step instructions into pipeline notation."""
        transforms: list[str] = []

        # Check if text has sequential pattern
        has_sequence = sum(1 for p in PIPE_KEYWORDS if re.search(p, text))
        if has_sequence < 2:
            return text, transforms

        # Extract steps
        steps: list[str] = []
        for pattern in SEQUENCE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                step = match.group(1).strip().rstrip(".,;")
                steps.append(step)

        if len(steps) >= 2:
            pipeline = "PIPE(" + " → ".join(steps) + ")"
            # Replace the sequential text with the pipeline
            # Find the span from first to last step
            first_match = re.search(SEQUENCE_PATTERNS[0], text)
            last_pattern = SEQUENCE_PATTERNS[min(len(steps), len(SEQUENCE_PATTERNS)) - 1]
            last_match = re.search(last_pattern, text)

            if first_match and last_match:
                text = text[:first_match.start()] + pipeline + text[last_match.end():]
                transforms.append(f"collapsed {len(steps)} steps into pipeline")

        return text, transforms

    def reset_index(self) -> None:
        """Reset the semantic index (e.g., new conversation)."""
        self._index = SemanticIndex()
