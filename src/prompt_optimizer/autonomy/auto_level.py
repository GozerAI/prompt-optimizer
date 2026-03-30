"""Autonomous optimization level selection based on input characteristics.

Analyzes input text for length, complexity, token count, and structural
patterns to automatically select the best compression layer (1-3) and
fidelity threshold without manual tuning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from prompt_optimizer.memory_tokenizer import count_tokens_efficient


@dataclass
class InputProfile:
    """Characterization of an input prompt."""

    token_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    has_agent_codes: bool = False
    has_parameters: bool = False
    has_numbers: bool = False
    has_structured_syntax: bool = False
    complexity_score: float = 0.0  # 0.0 (trivial) to 1.0 (very complex)
    filler_ratio: float = 0.0  # proportion of filler/hedging words


@dataclass
class LevelRecommendation:
    """Recommended optimization settings for a given input."""

    max_layer: int = 2
    min_fidelity: float = 0.50
    target_reduction: float | None = None
    confidence: float = 0.0
    reasoning: list[str] = field(default_factory=list)


# Filler words that indicate L1 structural compression will be effective.
_FILLER_WORDS = frozenset({
    "please", "kindly", "could", "would", "maybe", "perhaps",
    "basically", "essentially", "fundamentally", "hello", "hey",
    "dear", "thanks", "thank", "regards", "opinion", "believe",
    "think", "seems", "appears", "mentioned", "earlier", "previously",
})

# Agent codes indicating structured directive language.
_AGENT_PATTERN = re.compile(r"@[A-Z]{2,5}")
_PARAM_PATTERN = re.compile(r"\{[^}]+\}")
_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?%?")
_SENTENCE_SPLIT = re.compile(r"[.\!?]+\s+")
