"""Layer 1: Structural compression — strip natural language into typed envelopes."""

from __future__ import annotations

import re
from typing import Optional

from prompt_optimizer.envelope import TypedEnvelope
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

    def compress(self, text: str, context: CompressionContext) -> LayerResult:
        input_tokens = count_tokens(text)
        transformations: list[str] = []

        # Step 1: Extract envelope from ORIGINAL text (before stripping)
        # Only extract envelope for single-action prompts; multi-step is L2's job
        sentence_count = len([s for s in re.split(r"[.!?]\s+", text) if s.strip()])
        action_count = sum(1 for p, _ in ACTION_PATTERNS if re.search(p, text))
        envelope = None
        if sentence_count <= 2 or action_count <= 1:
            envelope = self._extract_envelope(text, context)

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
        if envelope:
            output = envelope.to_compact()
            transformations.append("converted to envelope")
        else:
            output = cleaned
            transformations.append("filler stripped only (no envelope extracted)")

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
        """
        # If it looks like an envelope, expand it
        if compressed.startswith("@") or any(
            compressed.startswith(a) for a in ("ANALYZE", "GENERATE", "DECIDE", "SUMMARIZE",
                                                "DELEGATE", "RECOMMEND", "FORECAST", "REPORT",
                                                "PLAN", "MONITOR", "OPTIMIZE", "ASSESS",
                                                "EVALUATE", "REVIEW")
        ):
            return self._expand_compact(compressed)
        return compressed

    def _extract_envelope(self, text: str, context: CompressionContext) -> Optional[TypedEnvelope]:
        """Try to extract a TypedEnvelope from cleaned text."""
        action = None
        for pattern, action_name in ACTION_PATTERNS:
            if re.search(pattern, text):
                action = action_name
                break

        if not action:
            return None

        # Extract recipient
        recipient = None
        for pattern in AGENT_PATTERNS:
            match = re.search(pattern, text)
            if match:
                recipient = match.group(1).upper()
                break

        # Also check context for agent codes
        if not recipient and context.agent_codes:
            for code in context.agent_codes:
                if re.search(rf"\b{re.escape(code)}\b", text, re.IGNORECASE):
                    recipient = code.upper()
                    break

        # Extract target: the main noun/object after the action verb
        target = self._extract_target(text, action)

        # Extract priority
        priority = None
        for pattern, level in PRIORITY_PATTERNS:
            if re.search(pattern, text):
                priority = level
                break

        # Extract modifiers
        modifiers: list[str] = []
        for pattern, mod in MODIFIER_PATTERNS:
            if re.search(pattern, text):
                modifiers.append(mod)

        # Extract constraints (sentences with "must", "should", "within", "limit")
        constraints = self._extract_constraints(text)

        # Extract params (key-value-like patterns)
        params = self._extract_params(text)

        # Extract response format hints
        response_format = self._extract_response_format(text)

        return TypedEnvelope(
            action=action,
            target=target or "unspecified",
            params=params,
            constraints=constraints,
            response_format=response_format,
            recipient=recipient,
            priority=priority,
            modifiers=modifiers,
        )

    def _extract_target(self, text: str, action: str) -> Optional[str]:
        """Extract the main object/target of the action."""
        # Look for "analyze X", "generate X report", etc.
        for pattern, action_name in ACTION_PATTERNS:
            if action_name == action:
                match = re.search(pattern + r"\s+(?:the\s+)?(.+?)(?:\s+and\s|\s+for\s|\s+with\s|[.,;]|$)", text)
                if match:
                    target = match.group(2) if match.lastindex and match.lastindex >= 2 else None
                    if not target:
                        # Try broader capture
                        after_verb = re.search(pattern + r"\s+(?:the\s+)?(\S+(?:\s+\S+){0,3})", text)
                        if after_verb:
                            target = after_verb.group(after_verb.lastindex or 1)
                    if target:
                        return target.strip().rstrip(".,;:!?")
        return None

    def _extract_constraints(self, text: str) -> list[str]:
        """Extract constraint phrases."""
        constraints: list[str] = []
        constraint_patterns = [
            r"(?i)\bmust\s+(.+?)(?:[.,;]|$)",
            r"(?i)\bshould\s+(.+?)(?:[.,;]|$)",
            r"(?i)\bwithin\s+(.+?)(?:[.,;]|$)",
            r"(?i)\blimit(?:ed)?\s+(?:to\s+)?(.+?)(?:[.,;]|$)",
            r"(?i)\bno more than\s+(.+?)(?:[.,;]|$)",
            r"(?i)\bat (?:most|least)\s+(.+?)(?:[.,;]|$)",
        ]
        for pattern in constraint_patterns:
            for match in re.finditer(pattern, text):
                constraints.append(match.group(1).strip())
        return constraints

    def _extract_params(self, text: str) -> dict[str, str]:
        """Extract key-value-like parameters."""
        params: dict[str, str] = {}

        # Period/time references
        time_match = re.search(r"(?i)\b(Q[1-4]\s*\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}|\d{4})", text)
        if time_match:
            params["period"] = time_match.group(1)

        # Comparison references
        compare_match = re.search(r"(?i)compar(?:e|ed|ing)\s+(?:to|with|against)\s+(.+?)(?:[.,;]|$)", text)
        if compare_match:
            params["compare"] = compare_match.group(1).strip()

        # Numeric values
        for match in re.finditer(r"(?i)(\w+)\s+(?:of|is|are|was|=)\s+(\$?[\d,.]+%?)", text):
            params[match.group(1).lower()] = match.group(2)

        return params

    def _extract_response_format(self, text: str) -> Optional[str]:
        """Extract response format hints."""
        format_patterns = [
            (r"(?i)\b(breakdown|itemized|line[- ]by[- ]line)\b", "breakdown"),
            (r"(?i)\b(summary|overview|highlights)\b", "summary"),
            (r"(?i)\b(report|analysis|assessment)\b", "report"),
            (r"(?i)\b(list|bullet points?|enumerat[es]*)\b", "list"),
            (r"(?i)\b(table|matrix|grid)\b", "table"),
            (r"(?i)\b(yes[/ ]no|binary|go[/ ]no[- ]go)\b", "decision"),
        ]
        for pattern, fmt in format_patterns:
            if re.search(pattern, text):
                return fmt
        return None

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
