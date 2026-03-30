"""Layer 3: Context compression — replace shared context with blackboard pointers."""

from __future__ import annotations

import re

from prompt_optimizer.layers.base import CompressionLayer
from prompt_optimizer.tokenizer import count_tokens
from prompt_optimizer.types import CompressionContext, LayerResult

# Context categories and their extraction patterns
CONTEXT_CATEGORIES = {
    "org": [
        r"(?i)(?:our|the)\s+(?:company|organization|firm|business)\s+\S+(?:\s+\S+){1,8}?(?=,|\.|;|$)",
        r"(?i)(?:the\s+)?board(?:'s)?\s+(?:mandate|directive|decision|requirement)\s+\S+(?:\s+\S+){1,8}?(?=,|\.|;|$)",
        r"(?i)(?:company|org(?:anization)?)\s+(?:policy|policies|guidelines?)\s+\S+(?:\s+\S+){1,6}?(?=,|\.|;|$)",
    ],
    "financial": [
        r"(?i)(?:revenue|profit|margin|cost|budget|earnings|income)\s+(?:of|is|are|was|were|at)\s+\S+(?:\s+\S+){0,5}?(?=,|\.|;|$)",
        r"(?i)\$[\d,.]+[MBKmk]?\b(?:\s+\S+){0,4}?(?=,|\.|;|$)",
        r"(?i)(?:Q[1-4]|quarterly|annual|monthly)\s+(?:revenue|results|numbers|figures)\s+\S+(?:\s+\S+){0,5}?(?=,|\.|;|$)",
    ],
    "strategic": [
        r"(?i)(?:pending|upcoming|planned)\s+(?:Series\s+[A-F]|IPO|merger|acquisition|partnership)\s+\S+(?:\s+\S+){0,6}?(?=,|\.|;|$)",
        r"(?i)(?:strategic|long[- ]term|growth)\s+(?:plan|initiative|goal|objective|direction)\s+\S+(?:\s+\S+){0,6}?(?=,|\.|;|$)",
    ],
    "technical": [
        r"(?i)(?:infrastructure|system|platform|architecture)\s+(?:costs?|requirements?|constraints?)\s+\S+(?:\s+\S+){0,6}?(?=,|\.|;|$)",
        r"(?i)(?:migration|deployment|upgrade|rollout)\s+\S+(?:\s+\S+){0,6}?(?=,|\.|;|$)",
    ],
    "historical": [
        r"(?i)(?:previously|historically|in the past|last\s+(?:quarter|year|month))\s+\S+(?:\s+\S+){0,6}?(?=,|\.|;|$)",
        r"(?i)(?:the\s+)?(?:CTO|CFO|CEO|COO|CIO|CMO)(?:'s)?\s+(?:assessment|analysis|report|recommendation)\s+(?:that|shows?|indicates?)\s+\S+(?:\s+\S+){0,8}?(?=,|\.|;|$)",
    ],
}


class ContextualLayer(CompressionLayer):
    """Layer 3: Context compression via blackboard pointers.

    Strips shared organizational context entirely and replaces with
    versioned pointers to a blackboard state store. Most aggressive
    compression with highest risk.
    """

    level = 3
    name = "contextual"
    risk_range = (0.10, 0.25)

    def compress(self, text: str, context: CompressionContext) -> LayerResult:
        input_tokens = count_tokens(text)
        transformations: list[str] = []

        if not context.blackboard:
            # Can't do L3 without a blackboard
            return LayerResult(
                layer=3,
                input_text=text,
                output_text=text,
                input_tokens=input_tokens,
                output_tokens=input_tokens,
                risk_score=0.0,
                transformations=["skipped: no blackboard available"],
                reversible=True,
            )

        result_text = text
        bb_refs: list[str] = []

        # Extract and store context by category
        for category, patterns in CONTEXT_CATEGORIES.items():
            for pattern in patterns:
                for match in re.finditer(pattern, result_text):
                    context_text = match.group(0).strip()
                    if len(context_text.split()) < 3:
                        continue  # Too short to bother

                    # Store in blackboard
                    pointer = context.blackboard.put(
                        namespace=category,
                        key=self._make_key(context_text),
                        value=context_text,
                    )
                    bb_refs.append(pointer)

                    # Replace in text
                    result_text = result_text.replace(context_text, f"[{pointer}]")
                    transformations.append(
                        f"externalized {category}: '{context_text[:40]}...' -> [{pointer}]"
                    )

        # Clean up multiple spaces and dangling punctuation
        result_text = re.sub(r"\s+", " ", result_text).strip()
        result_text = re.sub(r"\[\s*\]", "", result_text)

        # If we have refs, prepend them as a header
        if bb_refs:
            refs_header = "bb=[" + ",".join(bb_refs) + "]"
            result_text = f"{refs_header} {result_text}"

        output_tokens = count_tokens(result_text)

        # Risk scales with how many context blocks were externalized
        num_externalized = len(bb_refs)
        risk = self.risk_range[0] + min(num_externalized * 0.03, self.risk_range[1] - self.risk_range[0])

        return LayerResult(
            layer=3,
            input_text=text,
            output_text=result_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            risk_score=min(risk, self.risk_range[1]),
            transformations=transformations,
            reversible=True,
        )

    def decompress(self, compressed: str, context: CompressionContext) -> str:
        """Resolve blackboard pointers back to their values."""
        if not context.blackboard:
            return compressed

        result = compressed

        # Remove bb= header
        result = re.sub(r"bb=\[.+?\]\s*", "", result)

        # Resolve inline pointers (namespace:key@vN where key may contain underscores)
        for match in re.finditer(r"\[([\w:]+@v\d+)\]", result):
            pointer = match.group(1)
            if context.blackboard.has(pointer):
                value = context.blackboard.get(pointer)
                result = result.replace(match.group(0), str(value))

        return result

    def _make_key(self, text: str) -> str:
        """Generate a short key from context text."""
        # Use first few significant words
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        key_words = [w for w in words[:3] if w not in ("the", "and", "for", "with", "that", "this")]
        return "_".join(key_words) if key_words else "context"
