"""Reconstruction verifier — expands compressed prompts and checks for semantic drift."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from prompt_optimizer.blackboard import Blackboard
from prompt_optimizer.types import (
    CompressedPrompt,
    CompressionContext,
    DriftFlag,
    FidelityReport,
    LayerFidelity,
    Recommendation,
    Severity,
)


class ReconstructionVerifier:
    """Verifies that compressed prompts preserve original meaning."""

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        fact_threshold: float = 0.8,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.fact_threshold = fact_threshold

    def verify(
        self,
        original: str,
        compressed: CompressedPrompt,
        layers: list | None = None,
        context: CompressionContext | None = None,
    ) -> FidelityReport:
        """Verify fidelity of compression by reconstructing and comparing."""
        if context is None:
            context = CompressionContext()

        drift_flags: list[DriftFlag] = []
        per_layer: list[LayerFidelity] = []

        # Reconstruct through each layer in reverse
        reconstructed = compressed.compressed_text
        if layers:
            for layer in reversed(layers):
                reconstructed = layer.decompress(reconstructed, context)

        # Measure similarity
        similarity = self._text_similarity(original, reconstructed)

        # Extract and compare key facts
        original_facts = self._extract_facts(original)
        reconstructed_facts = self._extract_facts(reconstructed)
        fact_preservation = self._compare_facts(original_facts, reconstructed_facts, drift_flags)

        # Check for missing agent codes
        original_agents = set(re.findall(r"\b(CEO|COO|CTO|CFO|CIO|CMO|CHRO|CLO|CSO|CDO|CPO|CRO)\b", original, re.IGNORECASE))
        reconstructed_agents = set(re.findall(r"\b(CEO|COO|CTO|CFO|CIO|CMO|CHRO|CLO|CSO|CDO|CPO|CRO)\b", reconstructed, re.IGNORECASE))
        missing_agents = original_agents - reconstructed_agents
        if missing_agents:
            drift_flags.append(DriftFlag(
                layer=0,
                category="missing_fact",
                description=f"Agent codes lost: {missing_agents}",
                severity=Severity.ERROR,
            ))

        # Check for missing numbers
        original_numbers = set(re.findall(r"\$?[\d,.]+%?", original))
        reconstructed_numbers = set(re.findall(r"\$?[\d,.]+%?", reconstructed))
        missing_numbers = original_numbers - reconstructed_numbers
        if missing_numbers:
            drift_flags.append(DriftFlag(
                layer=0,
                category="altered_value",
                description=f"Numbers lost: {missing_numbers}",
                severity=Severity.WARNING,
            ))

        # Score per layer
        for layer_result in compressed.layer_results:
            completeness = fact_preservation
            accuracy = similarity
            actionability = self._score_actionability(original, compressed.compressed_text)

            per_layer.append(LayerFidelity(
                layer=layer_result.layer,
                completeness=completeness,
                accuracy=accuracy,
                actionability=actionability,
            ))

        # Overall score
        if per_layer:
            overall = sum(lf.overall for lf in per_layer) / len(per_layer)
        else:
            overall = similarity

        # Recommendation
        error_count = sum(1 for d in drift_flags if d.severity == Severity.ERROR)
        warning_count = sum(1 for d in drift_flags if d.severity == Severity.WARNING)

        if error_count > 0 or overall < 0.5:
            recommendation = Recommendation.UNSAFE
        elif warning_count > 2 or overall < self.similarity_threshold:
            recommendation = Recommendation.REVIEW
        else:
            recommendation = Recommendation.SAFE

        return FidelityReport(
            overall_score=overall,
            per_layer=per_layer,
            drift_flags=drift_flags,
            recommendation=recommendation,
        )

    def _text_similarity(self, a: str, b: str) -> float:
        """Normalized text similarity using SequenceMatcher."""
        # Normalize both
        a_norm = re.sub(r"\s+", " ", a.lower().strip())
        b_norm = re.sub(r"\s+", " ", b.lower().strip())
        return SequenceMatcher(None, a_norm, b_norm).ratio()

    def _extract_facts(self, text: str) -> set[str]:
        """Extract key facts: named entities, numbers, agent codes, action verbs."""
        facts: set[str] = set()

        # Numbers with context
        for match in re.finditer(r"\$?[\d,.]+%?\s*[MBKmk]?\w*", text):
            facts.add(match.group(0).strip().lower())

        # Agent codes
        for match in re.finditer(r"\b(CEO|COO|CTO|CFO|CIO|CMO|CHRO|CLO|CSO|CDO|CPO|CRO)\b", text, re.IGNORECASE):
            facts.add(match.group(0).upper())

        # Action verbs
        for match in re.finditer(r"\b(analyz|generat|evaluat|decid|summariz|delegat|recommend|forecast|plan|monitor|optimiz)\w*\b", text, re.IGNORECASE):
            facts.add(match.group(0).lower())

        # Time references
        for match in re.finditer(r"\b(Q[1-4]\s*\d{4}|\d{4})\b", text):
            facts.add(match.group(0))

        return facts

    def _compare_facts(
        self, original: set[str], reconstructed: set[str], drift_flags: list[DriftFlag]
    ) -> float:
        """Compare fact sets, flag missing facts."""
        if not original:
            return 1.0

        missing = original - reconstructed
        for fact in missing:
            drift_flags.append(DriftFlag(
                layer=0,
                category="missing_fact",
                description=f"Fact not preserved: '{fact}'",
                severity=Severity.WARNING,
                original_fragment=fact,
            ))

        return len(original & reconstructed) / len(original)

    def _score_actionability(self, original: str, compressed: str) -> float:
        """Score whether the compressed version is equally actionable."""
        # Check if the action verb is preserved
        original_actions = set(re.findall(r"\b(analyz|generat|evaluat|decid|summariz|delegat|recommend|forecast)\w*\b", original, re.IGNORECASE))
        compressed_actions = set(re.findall(r"\b(analyz|generat|evaluat|decid|summariz|delegat|recommend|forecast|ANALYZE|GENERATE|EVALUATE|DECIDE|SUMMARIZE|DELEGATE|RECOMMEND|FORECAST)\w*\b", compressed))

        if not original_actions:
            return 1.0

        # Normalize for comparison
        orig_normalized = {a.lower()[:6] for a in original_actions}
        comp_normalized = {a.lower()[:6] for a in compressed_actions}

        return len(orig_normalized & comp_normalized) / len(orig_normalized)
