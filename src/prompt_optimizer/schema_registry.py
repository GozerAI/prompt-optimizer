"""Schema registry — shared agent vocabularies and abbreviation maps."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EnvelopeSchema:
    """Expected parameter schema for an envelope action."""

    action: str
    required_params: list[str] = field(default_factory=list)
    optional_params: list[str] = field(default_factory=list)
    description: str = ""


class SchemaRegistry:
    """Shared vocabulary for agent communication abbreviations."""

    # Default executive codes
    DEFAULT_AGENT_CODES = {
        "Chief Executive Officer": "CEO",
        "Chief Operating Officer": "COO",
        "Chief Technology Officer": "CTO",
        "Chief Financial Officer": "CFO",
        "Chief Information Officer": "CIO",
        "Chief Marketing Officer": "CMO",
        "Chief Human Resources Officer": "CHRO",
        "Chief Legal Officer": "CLO",
        "Chief Security Officer": "CSO",
        "Chief Data Officer": "CDO",
        "Chief Product Officer": "CPO",
        "Chief Revenue Officer": "CRO",
        "Chief Strategy Officer": "CSTRO",
        "Chief Innovation Officer": "CINO",
        "Chief Risk Officer": "CRISKO",
        "Chief Sustainability Officer": "CSUSO",
    }

    # Standard action verbs
    STANDARD_ACTIONS = {
        "analyze": "ANALYZE",
        "generate": "GENERATE",
        "evaluate": "EVALUATE",
        "delegate": "DELEGATE",
        "decide": "DECIDE",
        "summarize": "SUMMARIZE",
        "assess": "ASSESS",
        "recommend": "RECOMMEND",
        "forecast": "FORECAST",
        "report": "REPORT",
        "review": "REVIEW",
        "plan": "PLAN",
        "monitor": "MONITOR",
        "optimize": "OPTIMIZE",
    }

    def __init__(self) -> None:
        self._vocabularies: dict[str, dict[str, str]] = {"default": {}}
        self._reverse_maps: dict[str, dict[str, str]] = {"default": {}}
        self._envelope_schemas: dict[str, EnvelopeSchema] = {}

        # Load defaults
        self.register_vocabulary("agents", self.DEFAULT_AGENT_CODES)
        self.register_vocabulary("actions", self.STANDARD_ACTIONS)

    def register_vocabulary(self, domain: str, mappings: dict[str, str]) -> None:
        """Register abbreviation mappings for a domain."""
        if domain not in self._vocabularies:
            self._vocabularies[domain] = {}
            self._reverse_maps[domain] = {}

        self._vocabularies[domain].update(mappings)
        self._reverse_maps[domain].update({v: k for k, v in mappings.items()})

    def abbreviate(self, text: str) -> str:
        """Replace known full forms with abbreviations."""
        result = text
        for domain_map in self._vocabularies.values():
            # Sort by length descending to match longer phrases first
            for full, abbrev in sorted(domain_map.items(), key=lambda x: -len(x[0])):
                result = re.sub(re.escape(full), abbrev, result, flags=re.IGNORECASE)
        return result

    def expand(self, abbreviated: str) -> str:
        """Replace abbreviations with full forms."""
        result = abbreviated
        for domain_map in self._reverse_maps.values():
            for abbrev, full in sorted(domain_map.items(), key=lambda x: -len(x[0])):
                # Only expand standalone abbreviations (word boundaries)
                result = re.sub(rf"\b{re.escape(abbrev)}\b", full, result)
        return result

    def register_envelope_schema(self, action: str, schema: EnvelopeSchema) -> None:
        """Register expected parameter schema for an action."""
        self._envelope_schemas[action] = schema

    def validate_envelope(self, envelope: Any) -> list[str]:
        """Validate an envelope against registered schemas. Returns error list."""
        errors: list[str] = []

        if envelope.action not in self._envelope_schemas:
            return errors  # No schema registered, skip validation

        schema = self._envelope_schemas[envelope.action]
        for param in schema.required_params:
            if param not in envelope.params:
                errors.append(f"Missing required param '{param}' for action '{envelope.action}'")

        return errors

    def get_action(self, verb: str) -> Optional[str]:
        """Look up a standard action verb."""
        lower = verb.lower()
        if lower in self.STANDARD_ACTIONS:
            return self.STANDARD_ACTIONS[lower]
        # Check reverse
        upper = verb.upper()
        for k, v in self.STANDARD_ACTIONS.items():
            if v == upper:
                return v
        return None
