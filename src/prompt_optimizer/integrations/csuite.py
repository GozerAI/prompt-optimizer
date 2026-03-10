"""C-Suite integration adapter for prompt-optimizer.

Wraps the optimizer for use with C-Suite's AgentMessage and
communicator abstractions. Install with: pip install prompt-optimizer[csuite]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from prompt_optimizer.blackboard import Blackboard
from prompt_optimizer.progressive import ProgressiveOptimizer
from prompt_optimizer.schema_registry import SchemaRegistry
from prompt_optimizer.types import CompressedPrompt, CompressionContext

if TYPE_CHECKING:
    pass  # C-Suite types would be imported here


class CSuitePromptOptimizer:
    """Adapter that wraps prompt-optimizer for C-Suite's communication layer."""

    # Standard C-Suite executive codes
    EXECUTIVE_CODES = [
        "CEO", "COO", "CTO", "CFO", "CIO", "CMO",
        "CHRO", "CLO", "CSO", "CDO", "CPO", "CRO",
        "CSTRO", "CINO", "CRISKO", "CSUSO",
    ]

    def __init__(
        self,
        optimizer: ProgressiveOptimizer | None = None,
        blackboard: Blackboard | None = None,
        schema_registry: SchemaRegistry | None = None,
        default_max_layer: int = 2,
    ) -> None:
        self.optimizer = optimizer or ProgressiveOptimizer()
        self.blackboard = blackboard or Blackboard()
        self.schema_registry = schema_registry or SchemaRegistry()
        self.default_max_layer = default_max_layer

    def _build_context(
        self,
        history: list[str] | None = None,
        org_context: dict[str, Any] | None = None,
    ) -> CompressionContext:
        """Build a CompressionContext from C-Suite state."""
        ctx = CompressionContext(
            agent_codes=self.EXECUTIVE_CODES,
            blackboard=self.blackboard,
            schema_registry=self.schema_registry,
            history=history or [],
        )

        # Store org context in blackboard if provided
        if org_context:
            for key, value in org_context.items():
                self.blackboard.put("org", key, value)

        return ctx

    def optimize_message(
        self,
        payload: str,
        sender: str | None = None,
        recipient: str | None = None,
        history: list[str] | None = None,
        max_layer: int | None = None,
    ) -> CompressedPrompt:
        """Compress a message payload.

        This is the primary interface for optimizing inter-executive messages.
        """
        context = self._build_context(history=history)
        return self.optimizer.optimize(
            text=payload,
            context=context,
            max_layer=max_layer or self.default_max_layer,
        )

    def optimize_task_prompt(
        self,
        task_description: str,
        agent_code: str,
        org_context: dict[str, Any] | None = None,
        history: list[str] | None = None,
        max_layer: int | None = None,
    ) -> CompressedPrompt:
        """Compress the prompt built by CSuiteAgent.reason().

        Call this before llm_provider.generate() to reduce token usage.
        """
        context = self._build_context(history=history, org_context=org_context)
        return self.optimizer.optimize(
            text=task_description,
            context=context,
            max_layer=max_layer or self.default_max_layer,
        )

    def restore(self, compressed: CompressedPrompt) -> str:
        """Decompress back to natural language (for debugging/display)."""
        text = compressed.compressed_text
        context = self._build_context()

        for layer in reversed(self.optimizer.layers):
            if layer.level in compressed.layers_applied:
                text = layer.decompress(text, context)

        return text

    def store_org_context(self, key: str, value: Any) -> str:
        """Store organizational context in the blackboard for L3 compression."""
        return self.blackboard.put("org", key, value)

    def store_agent_context(self, agent_code: str, key: str, value: Any) -> str:
        """Store agent-specific context."""
        return self.blackboard.put(f"agent:{agent_code}", key, value)
