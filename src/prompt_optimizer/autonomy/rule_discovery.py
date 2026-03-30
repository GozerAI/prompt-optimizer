"""Discover grammar rules from usage patterns."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DiscoveredRule:
    """A pattern discovered from usage data."""
    pattern: str
    replacement: str
    frequency: int
    confidence: float
    category: str = "phrase"
    examples: list[str] = field(default_factory=list)


@dataclass
class DiscoveryReport:
    """Results of a rule discovery pass."""
    rules: list[DiscoveredRule] = field(default_factory=list)
    corpus_size: int = 0
    unique_patterns: int = 0

    @property
    def high_confidence_rules(self) -> list[DiscoveredRule]:
        return [r for r in self.rules if r.confidence >= 0.7]


class RuleDiscovery:
    """Discovers compression rules from a corpus of optimization examples."""

    def __init__(self, *, min_frequency: int = 3, min_confidence: float = 0.5, max_rules: int = 100) -> None:
        self._min_frequency = min_frequency
        self._min_confidence = min_confidence
        self._max_rules = max_rules
        self._corpus: list[tuple[str, str]] = []

    def add_example(self, original: str, compressed: str) -> None:
        self._corpus.append((original, compressed))

    def add_examples(self, pairs: list[tuple[str, str]]) -> None:
        self._corpus.extend(pairs)

    @property
    def corpus_size(self) -> int:
        return len(self._corpus)

    def discover(self) -> DiscoveryReport:
        if not self._corpus:
            return DiscoveryReport()
        rules: list[DiscoveredRule] = []
        rules.extend(self._discover_phrase_patterns())
        rules.extend(self._discover_agent_patterns())
        rules.extend(self._discover_param_patterns())
        rules.extend(self._discover_structural_patterns())
        rules = [r for r in rules if r.frequency >= self._min_frequency and r.confidence >= self._min_confidence]
        rules.sort(key=lambda r: r.confidence * r.frequency, reverse=True)
        rules = rules[:self._max_rules]
        return DiscoveryReport(rules=rules, corpus_size=len(self._corpus), unique_patterns=len(rules))

    def _discover_phrase_patterns(self) -> list[DiscoveredRule]:
        phrase_counts: Counter[str] = Counter()
        phrase_examples: dict[str, list[str]] = {}
        for original, compressed in self._corpus:
            words = original.lower().split()
            comp_lower = compressed.lower()
            for n in (2, 3, 4):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i:i + n])
                    if phrase not in comp_lower and len(phrase) > 5:
                        phrase_counts[phrase] += 1
                        if phrase not in phrase_examples:
                            phrase_examples[phrase] = []
                        if len(phrase_examples[phrase]) < 3:
                            phrase_examples[phrase].append(original[:80])
        rules = []
        for phrase, count in phrase_counts.most_common(30):
            if count >= self._min_frequency:
                confidence = min(1.0, count / max(5, len(self._corpus) * 0.1))
                rules.append(DiscoveredRule(
                    pattern=re.escape(phrase), replacement="", frequency=count,
                    confidence=round(confidence, 2), category="phrase",
                    examples=phrase_examples.get(phrase, []),
                ))
        return rules

    def _discover_agent_patterns(self) -> list[DiscoveredRule]:
        counts: Counter[str] = Counter()
        pat = r"@([A-Z]{2,5})\s+([A-Z]+)"
        for _, compressed in self._corpus:
            for m in re.finditer(pat, compressed):
                counts["@{} {}".format(m.group(1), m.group(2))] += 1
        rules = []
        for combo, count in counts.most_common(20):
            if count >= self._min_frequency:
                parts = combo.split()
                shorthand = parts[0][:3] + parts[1][:3].lower() if len(parts) == 2 else combo[:6]
                confidence = min(1.0, count / max(3, len(self._corpus) * 0.05))
                rules.append(DiscoveredRule(
                    pattern=re.escape(combo), replacement=shorthand,
                    frequency=count, confidence=round(confidence, 2), category="agent_shorthand",
                ))
        return rules

    def _discover_param_patterns(self) -> list[DiscoveredRule]:
        counts: Counter[str] = Counter()
        pat = r"\{([^}]+)\}"
        for _, compressed in self._corpus:
            for m in re.finditer(pat, compressed):
                for kv in m.group(1).split(","):
                    kv = kv.strip()
                    if "=" in kv:
                        counts[kv.split("=")[0].strip()] += 1
        rules = []
        for key, count in counts.most_common(20):
            if count >= self._min_frequency and len(key) > 2:
                confidence = min(1.0, count / max(3, len(self._corpus) * 0.05))
                rules.append(DiscoveredRule(
                    pattern=key, replacement=key[:3] if len(key) > 3 else key,
                    frequency=count, confidence=round(confidence, 2), category="param_pattern",
                ))
        return rules

    def _discover_structural_patterns(self) -> list[DiscoveredRule]:
        counts: Counter[str] = Counter()
        examples: dict[str, list[str]] = {}
        for _, compressed in self._corpus:
            s = re.sub(r"@[A-Z]+", "@X", compressed)
            s = re.sub(r"\{[^}]*\}", "{...}", s)
            s = re.sub(r"\[[^\]]*\]", "[...]", s)
            s = re.sub(r"[a-z]+", "_", s)
            s = re.sub(r"\s+", " ", s).strip()
            if len(s) > 5:
                counts[s] += 1
                if s not in examples:
                    examples[s] = []
                if len(examples[s]) < 3:
                    examples[s].append(compressed[:80])
        rules = []
        for skel, count in counts.most_common(15):
            if count >= self._min_frequency:
                confidence = min(1.0, count / max(3, len(self._corpus) * 0.1))
                rules.append(DiscoveredRule(
                    pattern=skel, replacement="TMPL_{}".format(len(rules)),
                    frequency=count, confidence=round(confidence, 2), category="structure",
                    examples=examples.get(skel, []),
                ))
        return rules

    def clear_corpus(self) -> None:
        self._corpus.clear()
