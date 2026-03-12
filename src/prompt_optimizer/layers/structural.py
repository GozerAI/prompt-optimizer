"""Layer 1: Structural compression — strip natural language into typed envelopes.

Uses the grammar Compiler for NL → AST conversion and the Renderer for
AST → compact wire format. Falls back to filler-stripping only when the
Compiler can't extract structure.
"""

from __future__ import annotations

import re

from prompt_optimizer.grammar import Compiler, Lexer, Parser, Renderer
from prompt_optimizer.layers.base import CompressionLayer
from prompt_optimizer.tokenizer import count_tokens
from prompt_optimizer.types import CompressionContext, LayerResult

# Patterns that carry no information for agents
FILLER_PATTERNS = [
    # Greetings and politeness
    r"(?i)^(hey|hello|hi|dear)\s+\w+[,.]?\s*",
    r"(?i)\b(please|kindly|could you|would you|can you|I need you to|I'd like you to)\b\s*",
    r"(?i)\b(thank you|thanks|cheers|regards|best)\b[.,!]?\s*",
    # Hedging
    r"(?i)\b(I think|maybe|perhaps|it seems like|it appears that|I believe)\b\s*",
    r"(?i)\b(in my opinion|from my perspective|if you don't mind)\b[,.]?\s*",
    # Filler phrases
    r"(?i)\b(in order to|it is important to note that|as you may know)\b\s*",
    r"(?i)\b(basically|essentially|fundamentally|at the end of the day)\b[,.]?\s*",
    r"(?i)\b(as mentioned (earlier|before|previously|above))\b[,.]?\s*",
    r"(?i)\b(going forward|moving forward|from now on)\b[,.]?\s*",
    r"(?i)\b(take a (careful |close )?look at)\b\s*",
    r"(?i)\b(give me|provide me with)\s+(a\s+)?(detailed\s+)?",
    # Redundant connectors
    r"(?i)\b(additionally|furthermore|moreover|also|in addition)\b[,.]?\s*",
    r"(?i)\b(however|nevertheless|on the other hand)\b[,.]?\s*",
]

# Action verb extraction patterns
ACTION_PATTERNS = [
    (r"(?i)\b(analyz[es]*|assess|evaluat[es]*|examin[es]*|review)\b", "analyze"),
    (r"(?i)\b(look\s+at|breakdown|break\s+down|inspect)\b", "analyze"),
    (r"(?i)\b(generat[es]*|creat[es]*|produc[es]*|build|draft|write)\b", "generate"),
    (r"(?i)\b(decid[es]*|determin[es]*|choose|select|pick)\b", "decide"),
    (r"(?i)\b(summariz[es]*|condense|distill|recap)\b", "summarize"),
    (r"(?i)\b(delegat[es]*|assign|hand off|pass to|forward to)\b", "delegate"),
    (r"(?i)\b(recommend|suggest|propos[es]*|advis[es]*)\b", "recommend"),
    (r"(?i)\b(forecast|predict|project|estimat[es]*)\b", "forecast"),
    (r"(?i)\b(report|update|inform|notify|brief)\b", "report"),
    (r"(?i)\b(plan|schedul[es]*|organiz[es]*|coordinat[es]*)\b", "plan"),
    (r"(?i)\b(monitor|track|watch|observe|check)\b", "monitor"),
    (r"(?i)\b(optimiz[es]*|improv[es]*|enhanc[es]*|refin[es]*)\b", "optimize"),
]

# Agent code patterns
AGENT_PATTERNS = [
    r"(?i)\b(CEO|COO|CTO|CFO|CIO|CMO|CHRO|CLO|CSO|CDO|CPO|CRO)\b",
    r"(?i)\b(?:the\s+)?Chief\s+(\w+)\s+Officer\b",
]

# Priority/urgency markers
PRIORITY_PATTERNS = [
    (r"(?i)\b(ASAP|immediately|urgent(ly)?|right away|top priority)\b", "urgent"),
    (r"(?i)\b(high priority|important|critical)\b", "high"),
    (r"(?i)\b(when you (get a chance|can)|low priority|no rush)\b", "low"),
]

# Autonomy/discretion markers
MODIFIER_PATTERNS = [
    (r"(?i)\b(use your (judgment|discretion|expertise))\b", "discretion"),
    (r"(?i)\b(you think are relevant|as you see fit)\b", "discretion"),
    (r"(?i)\b(thorough(ly)?|in[- ]depth|comprehensive(ly)?|detail(ed)?)\b", "thorough"),
    (r"(?i)\b(brief(ly)?|quick(ly)?|high[- ]level|overview)\b", "brief"),
]


class StructuralLayer(CompressionLayer):
    """Layer 1: Rule-based structural compression.

    Strips politeness, filler, redundancy. Extracts action, target, params
    into a TypedEnvelope. Deterministic and fully reversible.
    """

    level = 1
    name = "structural"
    risk_range = (0.01, 0.05)

    def __init__(self, extra_filler_patterns: list[str] | None = None) -> None:
        self._extra_patterns = extra_filler_patterns or []
        self._compiler = Compiler()
        self._renderer = Renderer()
        self._lexer = Lexer()

    def compress(self, text: str, context: CompressionContext) -> LayerResult:
        input_tokens = count_tokens(text)
        transformations: list[str] = []

        # Step 1: Try Compiler on ORIGINAL text (before stripping)
        # Only compile single-action prompts; multi-step pipelines are L2's job
        sentence_count = len([s for s in re.split(r"[.!?]\s+", text) if s.strip()])
        action_count = sum(1 for p, _ in ACTION_PATTERNS if re.search(p, text))
        ast = None
        if sentence_count <= 2 or action_count <= 1:
            ast = self._compiler.compile(text)

        # Step 2: Strip filler
        cleaned = text
        all_patterns = FILLER_PATTERNS + self._extra_patterns
        for pattern in all_patterns:
            before = cleaned
            cleaned = re.sub(pattern, " ", cleaned)
            if cleaned != before:
                transformations.append(f"stripped: {pattern[:40]}")

        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Step 3: Generate compact output
        if ast:
            output = self._renderer.render(ast)
            transformations.append("compiled to AST and rendered")
        else:
            output = cleaned
            transformations.append("filler stripped only (no AST extracted)")

        output_tokens = count_tokens(output)

        result = LayerResult(
            layer=1,
            input_text=text,
            output_text=output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            risk_score=self.risk_range[0],
            transformations=transformations,
            reversible=True,
        )
        return result

    def decompress(self, compressed: str, context: CompressionContext) -> str:
        """Reconstruct natural language from compact form.

        Since L1 is lossy on filler (intentionally), decompression
        produces a clean version without restoring filler.
        Uses the grammar Parser → Renderer.render_human() pipeline.
        """
        # If it looks like compact grammar notation, parse and render as human-readable
        if compressed.startswith("@") or any(
            compressed.startswith(a) for a in ("ANALYZE", "GENERATE", "DECIDE", "SUMMARIZE",
                                                "DELEGATE", "RECOMMEND", "FORECAST", "REPORT",
                                                "PLAN", "MONITOR", "OPTIMIZE", "ASSESS",
                                                "EVALUATE", "REVIEW", "COST", "APPROVE")
        ):
            try:
                tokens = self._lexer.tokenize(compressed)
                ast = Parser(tokens).parse()
                return self._renderer.render_human(ast)
            except Exception:
                return self._expand_compact(compressed)
        return compressed

    def _expand_compact(self, compact: str) -> str:
        """Expand compact envelope notation back to readable text."""
        parts: list[str] = []

        # Parse recipient
        recipient_match = re.match(r"@(\w+)\s*", compact)
        if recipient_match:
            parts.append(f"To {recipient_match.group(1)}:")
            compact = compact[recipient_match.end():]

        # Parse action + target
        tokens = compact.split(None, 2)
        if tokens:
            action = tokens[0].lower()
            target = tokens[1] if len(tokens) > 1 else ""
            parts.append(f"{action.capitalize()} {target}")

        return " ".join(parts) if parts else compact
