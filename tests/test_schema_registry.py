"""Tests for SchemaRegistry."""

from prompt_optimizer.envelope import TypedEnvelope
from prompt_optimizer.schema_registry import EnvelopeSchema, SchemaRegistry


class TestSchemaRegistry:
    def setup_method(self):
        self.registry = SchemaRegistry()

    def test_default_agent_codes(self):
        result = self.registry.abbreviate("Ask the Chief Technology Officer")
        assert "CTO" in result

    def test_abbreviate_preserves_unknown(self):
        result = self.registry.abbreviate("Ask John about the project")
        assert "John" in result

    def test_expand(self):
        result = self.registry.expand("CTO should analyze")
        assert "Chief Technology Officer" in result

    def test_roundtrip(self):
        original = "Chief Financial Officer"
        abbreviated = self.registry.abbreviate(original)
        expanded = self.registry.expand(abbreviated)
        assert expanded == original

    def test_custom_vocabulary(self):
        self.registry.register_vocabulary("domain", {"market analysis": "MA"})
        result = self.registry.abbreviate("Run a market analysis")
        assert "MA" in result

    def test_get_action(self):
        assert self.registry.get_action("analyze") == "ANALYZE"
        assert self.registry.get_action("ANALYZE") == "ANALYZE"
        assert self.registry.get_action("unknown_verb") is None

    def test_envelope_validation_no_schema(self):
        env = TypedEnvelope(action="analyze", target="revenue")
        errors = self.registry.validate_envelope(env)
        assert errors == []  # No schema registered, no errors

    def test_envelope_validation_missing_param(self):
        self.registry.register_envelope_schema(
            "analyze",
            EnvelopeSchema(action="analyze", required_params=["period"]),
        )
        env = TypedEnvelope(action="analyze", target="revenue")
        errors = self.registry.validate_envelope(env)
        assert len(errors) == 1
        assert "period" in errors[0]

    def test_envelope_validation_passes(self):
        self.registry.register_envelope_schema(
            "analyze",
            EnvelopeSchema(action="analyze", required_params=["period"]),
        )
        env = TypedEnvelope(action="analyze", target="revenue", params={"period": "Q1-2026"})
        errors = self.registry.validate_envelope(env)
        assert errors == []

    def test_case_insensitive_abbreviation(self):
        result = self.registry.abbreviate("chief technology officer is here")
        assert "CTO" in result
