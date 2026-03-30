"""Self-documenting optimization passes.

Generates markdown documentation from optimization pass metadata including
name, description, expected token reduction, and risk level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PassMetadata:
    """Metadata for a single optimization pass."""

    name: str
    level: int
    description: str
    expected_reduction: tuple[float, float]
    risk_range: tuple[float, float]
    reversible: bool
    transformations: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class PassDocumenter:
    """Generates documentation from optimization pass metadata."""

    _LAYER_DOCS: dict[str, dict[str, Any]] = {
        "structural": {
            "description": (
                "Strips natural language into typed envelopes. Removes filler words, "
                "hedging, greetings, and politeness. Extracts action verbs and parameters "
                "into compact grammar notation. Uses the grammar Compiler for NL to AST "
                "conversion and the Renderer for AST to compact wire format."
            ),
            "expected_reduction": (0.55, 0.70),
            "reversible": True,
            "transformations": [
                "Filler word removal",
                "Greeting/politeness stripping",
                "Hedging phrase removal",
                "Action verb extraction",
                "NL to grammar AST compilation",
                "AST to compact wire format rendering",
            ],
            "dependencies": [],
        },
        "semantic": {
            "description": (
                "Deduplicates context blocks, resolves references, and applies pipeline "
                "shorthand notation. Uses content hashing for deduplication and replaces "
                "repeated context with reference IDs."
            ),
            "expected_reduction": (0.70, 0.80),
            "reversible": True,
            "transformations": [
                "Context block deduplication via content hashing",
                "Reference ID replacement for repeated content",
                "Pipeline pattern detection",
                "Sequential operation pipe notation",
                "Schema abbreviation",
            ],
            "dependencies": ["structural"],
        },
        "contextual": {
            "description": (
                "Replaces shared organizational context with versioned blackboard pointers. "
                "Extracts context by category (organizational, financial, strategic, "
                "technical, historical) and stores on a shared blackboard."
            ),
            "expected_reduction": (0.85, 0.95),
            "reversible": False,
            "transformations": [
                "Organizational context extraction",
                "Financial context extraction",
                "Strategic context extraction",
                "Technical context extraction",
                "Historical context extraction",
                "Blackboard pointer replacement (bb:ns:key@vN)",
            ],
            "dependencies": ["structural", "semantic"],
        },
    }

    def collect_pass_metadata(self) -> list[PassMetadata]:
        """Collect metadata from all registered optimization passes."""
        passes: list[PassMetadata] = []
        try:
            from prompt_optimizer.layers.contextual import ContextualLayer
            from prompt_optimizer.layers.semantic import SemanticLayer
            from prompt_optimizer.layers.structural import StructuralLayer
            for layer_cls in [StructuralLayer, SemanticLayer, ContextualLayer]:
                passes.append(self._extract_layer_metadata(layer_cls))
        except ImportError:
            pass
        return passes

    def _extract_layer_metadata(self, layer_cls: type) -> PassMetadata:
        """Extract metadata from a CompressionLayer subclass."""
        name = getattr(layer_cls, "name", layer_cls.__name__.lower())
        level = getattr(layer_cls, "level", 0)
        risk_range = getattr(layer_cls, "risk_range", (0.0, 1.0))
        docs = self._LAYER_DOCS.get(name, {})
        return PassMetadata(
            name=name, level=level,
            description=docs.get("description", layer_cls.__doc__ or "No description."),
            expected_reduction=docs.get("expected_reduction", (0.0, 1.0)),
            risk_range=risk_range,
            reversible=docs.get("reversible", True),
            transformations=docs.get("transformations", []),
            dependencies=docs.get("dependencies", []),
        )

    def generate_markdown(self, passes: list[PassMetadata] | None = None) -> str:
        """Generate comprehensive markdown documentation for all passes."""
        if passes is None:
            passes = self.collect_pass_metadata()
        lines: list[str] = []
        lines.append("# Prompt Optimizer -- Optimization Passes")
        lines.append("")
        lines.append("Auto-generated documentation from optimization pass metadata.")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append("| Layer | Name | Reduction | Risk | Reversible |")
        lines.append("|-------|------|-----------|------|------------|")
        for p in sorted(passes, key=lambda x: x.level):
            red = f"{p.expected_reduction[0]*100:.0f}%-{p.expected_reduction[1]*100:.0f}%"
            risk = f"{p.risk_range[0]*100:.0f}%-{p.risk_range[1]*100:.0f}%"
            rev = "Yes" if p.reversible else "No"
            lines.append(f"| L{p.level} | {p.name.title()} | {red} | {risk} | {rev} |")
        lines.append("")
        for p in sorted(passes, key=lambda x: x.level):
            lines.append(f"## Layer {p.level}: {p.name.title()}")
            lines.append("")
            lines.append(p.description)
            lines.append("")
            lines.append(f"- **Expected reduction**: {p.expected_reduction[0]*100:.0f}%-{p.expected_reduction[1]*100:.0f}%")
            lines.append(f"- **Risk range**: {p.risk_range[0]*100:.0f}%-{p.risk_range[1]*100:.0f}%")
            lines.append(f"- **Reversible**: {'Yes' if p.reversible else 'No'}")
            lines.append("")
            if p.transformations:
                lines.append("### Transformations")
                lines.append("")
                for t in p.transformations:
                    lines.append(f"- {t}")
                lines.append("")
            if p.dependencies:
                lines.append("### Dependencies")
                lines.append("")
                lines.append(f"Requires: {', '.join(d.title() for d in p.dependencies)}")
                lines.append("")
        return chr(10).join(lines)

    def generate_pass_summary(self, p: PassMetadata) -> str:
        """Generate a one-line summary for a single pass."""
        irr = "" if p.reversible else " (irreversible)"
        return (
            f"L{p.level} {p.name.title()}: "
            f"{p.expected_reduction[0]*100:.0f}-{p.expected_reduction[1]*100:.0f}% reduction, "
            f"{p.risk_range[0]*100:.0f}-{p.risk_range[1]*100:.0f}% risk{irr}"
        )


def generate_docs() -> str:
    """Generate full markdown docs for all optimization passes."""
    return PassDocumenter().generate_markdown()


def list_passes() -> list[PassMetadata]:
    """List all optimization pass metadata."""
    return PassDocumenter().collect_pass_metadata()
