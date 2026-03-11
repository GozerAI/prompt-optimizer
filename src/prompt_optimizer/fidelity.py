"""Fidelity scorer — measures information loss at each compression layer."""

from __future__ import annotations

import re

from prompt_optimizer.types import (
    CompressedPrompt,
    FidelityReport,
    LayerFidelity,
    Recommendation,
)


class FidelityScorer:
    """Scores information preservation across compression layers."""

    def score(self, original: str, layer_output: str, layer: int) -> LayerFidelity:
        """Score information preservation for a specific layer."""
        completeness = self._score_completeness(original, layer_output)
        accuracy = self._score_accuracy(original, layer_output)
        actionability = self._score_actionability(original, layer_output)

        return LayerFidelity(
            layer=layer,
            completeness=completeness,
            accuracy=accuracy,
            actionability=actionability,
        )

    def score_all(self, compressed: CompressedPrompt) -> FidelityReport:
        """Score across all applied layers."""
        per_layer: list[LayerFidelity] = []

        for result in compressed.layer_results:
            fidelity = self.score(result.input_text, result.output_text, result.layer)
            per_layer.append(fidelity)

        if per_layer:
            overall = sum(lf.overall for lf in per_layer) / len(per_layer)
        else:
            overall = 1.0

        if overall >= 0.85:
            recommendation = Recommendation.SAFE
        elif overall >= 0.65:
            recommendation = Recommendation.REVIEW
        else:
            recommendation = Recommendation.UNSAFE

        return FidelityReport(
            overall_score=overall,
            per_layer=per_layer,
            recommendation=recommendation,
        )

    def _score_completeness(self, original: str, compressed: str) -> float:
        """Are all key facts preserved?"""
        original_facts = self._extract_key_elements(original)
        compressed_facts = self._extract_key_elements(compressed)

        if not original_facts:
            return 1.0

        preserved = sum(1 for f in original_facts if self._fact_present(f, compressed_facts))
        return preserved / len(original_facts)

    def _score_accuracy(self, original: str, compressed: str) -> float:
        """Are preserved facts unchanged?

        Focuses on key elements (numbers, agent codes, time refs) rather than
        general word overlap, since structural compression intentionally
        transforms natural language into compact notation.
        """
        original_elements = self._extract_key_elements(original)
        compressed_elements = self._extract_key_elements(compressed)

        if not original_elements:
            # No key elements to compare — check content word preservation
            orig_tokens = set(re.findall(r"\b\w{4,}\b", original.lower()))
            comp_tokens = set(re.findall(r"\b\w{4,}\b", compressed.lower()))
            if not orig_tokens:
                return 1.0
            return len(orig_tokens & comp_tokens) / len(orig_tokens)

        preserved = sum(1 for e in original_elements if self._fact_present(e, compressed_elements))
        return preserved / len(original_elements)

    def _score_actionability(self, original: str, compressed: str) -> float:
        """Can the recipient execute the same action from compressed as from original?"""
        # Must preserve: action verb, target, and critical parameters
        original_actions = re.findall(
            r"\b(analyz|generat|evaluat|decid|summariz|delegat|recommend|forecast|plan|monitor|optimiz)\w*\b",
            original,
            re.IGNORECASE,
        )
        compressed_text = compressed.lower()

        if not original_actions:
            return 1.0

        preserved = sum(1 for a in original_actions if a.lower()[:5] in compressed_text)
        return preserved / len(original_actions)

    def _extract_key_elements(self, text: str) -> list[str]:
        """Extract key elements: numbers, agent codes, time refs, key nouns."""
        elements: list[str] = []

        # Numbers
        elements.extend(re.findall(r"\$?[\d,.]+%?", text))

        # Agent codes
        elements.extend(re.findall(r"\b(?:CEO|COO|CTO|CFO|CIO|CMO|CHRO|CLO)\b", text, re.IGNORECASE))

        # Time references
        elements.extend(re.findall(r"\bQ[1-4]\s*\d{4}\b", text))
        elements.extend(re.findall(r"\b\d{4}\b", text))

        return elements

    def _fact_present(self, fact: str, fact_set: list[str]) -> bool:
        """Check if a fact is present (with fuzzy matching for numbers)."""
        fact_lower = fact.lower().strip()
        for f in fact_set:
            if fact_lower == f.lower().strip():
                return True
            # Fuzzy match for numbers with different formatting
            if re.match(r"[\d$,.%]+", fact) and re.match(r"[\d$,.%]+", f):
                # Strip formatting and compare
                fact_digits = re.sub(r"[^\d.]", "", fact)
                f_digits = re.sub(r"[^\d.]", "", f)
                if fact_digits == f_digits:
                    return True
        return False
