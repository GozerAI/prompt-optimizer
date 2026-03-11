"""LLM-as-Judge fidelity evaluation for prompt compression.

Uses an LLM to evaluate whether compressed prompts preserve the original
intent and actionability. Opt-in, intended for offline calibration.

Requires: httpx (for Ollama HTTP client)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from prompt_optimizer.fidelity import FidelityScorer
from prompt_optimizer.types import CompressedPrompt, FidelityReport, LayerFidelity

FIDELITY_RUBRIC = """You are evaluating whether a compressed version of an inter-agent message \
preserves the original meaning sufficiently for the recipient agent to take the same action.

## Original message
{original}

## Compressed message
{compressed}

## Recipient agent
{recipient} ({recipient_role})

## Score on these dimensions (each 1-5):

1. **Intent Preservation**: Would the recipient understand the same goal/request?
   - 5: Identical intent. 4: Same intent, minor nuance lost. 3: Core intent preserved, \
some context missing. 2: Ambiguous intent. 1: Different intent.

2. **Context Sufficiency**: Does the compressed version provide enough context to act?
   - 5: All context preserved. 4: Key context preserved. 3: Adequate but gaps exist. \
2: Missing critical context. 1: Insufficient context.

3. **Actionability**: Would the recipient take the same concrete action?
   - 5: Identical action. 4: Same action, different approach. 3: Similar action. \
2: Different action. 1: Unable to act.

Respond with EXACTLY this JSON (no other text):
{{"intent_preservation": <1-5>, "context_sufficiency": <1-5>, "actionability": <1-5>, \
"reasoning": "<brief explanation>"}}"""


@dataclass
class LLMFidelityVerdict:
    """Result from LLM-as-judge fidelity evaluation."""

    intent_preservation: float  # 0-1
    context_sufficiency: float  # 0-1
    actionability: float  # 0-1
    reasoning: str = ""
    judge_model: str = ""
    error: Optional[str] = None

    @property
    def overall(self) -> float:
        """Weighted average matching FidelityScorer conventions."""
        return (
            self.intent_preservation * 0.4
            + self.context_sufficiency * 0.4
            + self.actionability * 0.2
        )


@dataclass
class CalibrationResult:
    """Result from comparing LLM scores vs rule-based scores."""

    samples: int
    mean_agreement: float  # 0-1, how closely scores match
    mean_llm_score: float
    mean_rule_score: float
    disagreements: list[dict[str, Any]] = field(default_factory=list)


class LLMFidelityScorer:
    """LLM-based fidelity evaluation using Ollama or compatible API.

    Usage:
        scorer = LLMFidelityScorer(base_url="http://localhost:11434")
        verdict = await scorer.score(original, compressed, "CFO", "financial analysis")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        judge_model: str = "qwen2.5:32b",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.judge_model = judge_model

    async def score(
        self,
        original: str,
        compressed: str,
        recipient: str = "agent",
        recipient_role: str = "general",
    ) -> LLMFidelityVerdict:
        """Score fidelity of compression using LLM judge."""
        prompt = FIDELITY_RUBRIC.format(
            original=original,
            compressed=compressed,
            recipient=recipient,
            recipient_role=recipient_role,
        )

        try:
            response_text = await self._call_llm(prompt)
            return self._parse_verdict(response_text)
        except Exception as e:
            return LLMFidelityVerdict(
                intent_preservation=0.0,
                context_sufficiency=0.0,
                actionability=0.0,
                error=str(e),
                judge_model=self.judge_model,
            )

    async def batch_calibrate(
        self,
        samples: list[tuple[str, str]],
        rule_scorer: FidelityScorer,
    ) -> CalibrationResult:
        """Compare LLM scores vs rule-based scores for calibration.

        Args:
            samples: List of (original, compressed) pairs.
            rule_scorer: Rule-based scorer to compare against.
        """
        agreements: list[float] = []
        llm_scores: list[float] = []
        rule_scores: list[float] = []
        disagreements: list[dict[str, Any]] = []

        for original, compressed in samples:
            llm_verdict = await self.score(original, compressed)
            rule_fidelity = rule_scorer.score(original, compressed, layer=1)

            if llm_verdict.error:
                continue

            llm_overall = llm_verdict.overall
            rule_overall = rule_fidelity.overall

            llm_scores.append(llm_overall)
            rule_scores.append(rule_overall)

            # Agreement = 1 - |difference|
            agreement = 1.0 - abs(llm_overall - rule_overall)
            agreements.append(agreement)

            if abs(llm_overall - rule_overall) > 0.2:
                disagreements.append({
                    "original": original[:100],
                    "compressed": compressed[:100],
                    "llm_score": llm_overall,
                    "rule_score": rule_overall,
                    "reasoning": llm_verdict.reasoning,
                })

        return CalibrationResult(
            samples=len(agreements),
            mean_agreement=sum(agreements) / len(agreements) if agreements else 0.0,
            mean_llm_score=sum(llm_scores) / len(llm_scores) if llm_scores else 0.0,
            mean_rule_score=sum(rule_scores) / len(rule_scores) if rule_scores else 0.0,
            disagreements=disagreements,
        )

    async def _call_llm(self, prompt: str) -> str:
        """Call Ollama API."""
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx is required for LLM fidelity scoring: pip install httpx")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.judge_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
            )
            response.raise_for_status()
            return response.json().get("response", "")

    def _parse_verdict(self, response: str) -> LLMFidelityVerdict:
        """Parse LLM response into a verdict."""
        # Extract JSON from response (may have surrounding text)
        json_match = re.search(r"\{[^{}]+\}", response)
        if not json_match:
            return LLMFidelityVerdict(
                intent_preservation=0.0,
                context_sufficiency=0.0,
                actionability=0.0,
                error=f"No JSON found in response: {response[:200]}",
                judge_model=self.judge_model,
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            return LLMFidelityVerdict(
                intent_preservation=0.0,
                context_sufficiency=0.0,
                actionability=0.0,
                error=f"Invalid JSON: {e}",
                judge_model=self.judge_model,
            )

        # Normalize 1-5 scores to 0-1
        def normalize(val: Any) -> float:
            try:
                return max(0.0, min(1.0, (float(val) - 1) / 4))
            except (ValueError, TypeError):
                return 0.0

        return LLMFidelityVerdict(
            intent_preservation=normalize(data.get("intent_preservation", 0)),
            context_sufficiency=normalize(data.get("context_sufficiency", 0)),
            actionability=normalize(data.get("actionability", 0)),
            reasoning=str(data.get("reasoning", "")),
            judge_model=self.judge_model,
        )


class CompositeFidelityScorer:
    """Uses rule-based scoring by default, optionally validates with LLM judge.

    The composite scorer always runs the rule-based scorer. When use_llm=True,
    it also runs the LLM scorer and blends the results (70% rule, 30% LLM).
    """

    def __init__(
        self,
        rule_scorer: FidelityScorer | None = None,
        llm_scorer: LLMFidelityScorer | None = None,
        use_llm: bool = False,
        llm_weight: float = 0.3,
    ) -> None:
        self.rule_scorer = rule_scorer or FidelityScorer()
        self.llm_scorer = llm_scorer
        self.use_llm = use_llm and llm_scorer is not None
        self.llm_weight = llm_weight

    def score(self, original: str, layer_output: str, layer: int) -> LayerFidelity:
        """Score using rule-based scorer (LLM scoring requires async)."""
        return self.rule_scorer.score(original, layer_output, layer)

    def score_all(self, compressed: CompressedPrompt) -> FidelityReport:
        """Score all layers using rule-based scorer."""
        return self.rule_scorer.score_all(compressed)

    async def score_with_llm(
        self,
        original: str,
        compressed: str,
        layer: int,
        recipient: str = "agent",
    ) -> LayerFidelity:
        """Score with both rule-based and LLM, blending results."""
        rule_fidelity = self.rule_scorer.score(original, compressed, layer)

        if not self.use_llm or not self.llm_scorer:
            return rule_fidelity

        verdict = await self.llm_scorer.score(original, compressed, recipient)
        if verdict.error:
            return rule_fidelity

        # Blend scores
        rw = 1.0 - self.llm_weight
        lw = self.llm_weight

        return LayerFidelity(
            layer=layer,
            completeness=rule_fidelity.completeness * rw + verdict.context_sufficiency * lw,
            accuracy=rule_fidelity.accuracy * rw + verdict.intent_preservation * lw,
            actionability=rule_fidelity.actionability * rw + verdict.actionability * lw,
        )
