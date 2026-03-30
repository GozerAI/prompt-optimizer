"""Typed envelope — structured message format replacing natural language.

Includes bridge methods to/from the grammar AST, allowing envelopes and
AST nodes to be used interchangeably.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from prompt_optimizer.grammar.ast_nodes import (
    ConstraintNode,
    ConstraintsNode,
    DirectiveNode,
    ModifierNode,
    OutputNode,
    ParamNode,
    ParamsNode,
    PriorityNode,
    RecipientNode,
)


@dataclass
class TypedEnvelope:
    """Structured representation of an agent communication."""

    action: str  # "analyze", "generate", "evaluate", "delegate", "decide"
    target: str  # what to act on
    params: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    context_refs: list[str] = field(default_factory=list)
    response_format: Optional[str] = None
    sender: Optional[str] = None
    recipient: Optional[str] = None
    priority: Optional[str] = None  # "low", "normal", "high", "urgent"
    modifiers: list[str] = field(default_factory=list)  # "discretion", "thorough", "brief"

    def to_compact(self) -> str:
        """Serialize to minimal wire format."""
        parts = [f"@{self.recipient}" if self.recipient else "", self.action.upper(), self.target]

        if self.params:
            param_str = " ".join(f"{k}={v}" for k, v in self.params.items())
            parts.append(f"{{{param_str}}}")

        if self.constraints:
            parts.append(f"[{', '.join(self.constraints)}]")

        if self.context_refs:
            parts.append(f"bb=[{','.join(self.context_refs)}]")

        if self.response_format:
            parts.append(f"-> {self.response_format}")

        if self.priority and self.priority != "normal":
            parts.append(f"!{self.priority}")

        if self.modifiers:
            parts.append(f"~{'~'.join(self.modifiers)}")

        return " ".join(p for p in parts if p)

    def to_dict(self) -> dict[str, Any]:
        """Full dict serialization."""
        d: dict[str, Any] = {"action": self.action, "target": self.target}
        if self.params:
            d["params"] = self.params
        if self.constraints:
            d["constraints"] = self.constraints
        if self.context_refs:
            d["context_refs"] = self.context_refs
        if self.response_format:
            d["response_format"] = self.response_format
        if self.sender:
            d["sender"] = self.sender
        if self.recipient:
            d["recipient"] = self.recipient
        if self.priority:
            d["priority"] = self.priority
        if self.modifiers:
            d["modifiers"] = self.modifiers
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TypedEnvelope:
        return cls(
            action=data["action"],
            target=data["target"],
            params=data.get("params", {}),
            constraints=data.get("constraints", []),
            context_refs=data.get("context_refs", []),
            response_format=data.get("response_format"),
            sender=data.get("sender"),
            recipient=data.get("recipient"),
            priority=data.get("priority"),
            modifiers=data.get("modifiers", []),
        )

    def to_ast(self) -> DirectiveNode:
        """Convert envelope to a grammar AST DirectiveNode."""
        recipient = RecipientNode(agent_code=self.recipient) if self.recipient else None

        params = None
        if self.params:
            params = ParamsNode(
                params=[ParamNode(key=k, value=str(v)) for k, v in self.params.items()]
            )

        constraints = None
        if self.constraints:
            constraints = ConstraintsNode(
                constraints=[ConstraintNode(text=c) for c in self.constraints]
            )

        output = OutputNode(format=self.response_format) if self.response_format else None
        priority = PriorityNode(level=self.priority) if self.priority and self.priority != "normal" else None
        modifiers = [ModifierNode(name=m) for m in self.modifiers]

        return DirectiveNode(
            action=self.action.upper(),
            target=self.target,
            recipient=recipient,
            params=params,
            constraints=constraints,
            output=output,
            priority=priority,
            modifiers=modifiers,
        )

    @classmethod
    def from_ast(cls, node: DirectiveNode) -> TypedEnvelope:
        """Create an envelope from a grammar AST DirectiveNode."""
        params: dict[str, Any] = {}
        if node.params:
            for p in node.params.params:
                params[p.key] = p.value if p.value is not None else True

        constraints: list[str] = []
        if node.constraints:
            constraints = [c.text for c in node.constraints.constraints]

        return cls(
            action=node.action.lower(),
            target=node.target,
            params=params,
            constraints=constraints,
            response_format=node.output.format if node.output else None,
            recipient=node.recipient.agent_code if node.recipient else None,
            priority=node.priority.level if node.priority else None,
            modifiers=[m.name for m in node.modifiers],
        )
