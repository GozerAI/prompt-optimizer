"""Tests for TypedEnvelope."""

from prompt_optimizer.envelope import TypedEnvelope


class TestTypedEnvelope:
    def test_to_compact_basic(self):
        env = TypedEnvelope(action="analyze", target="revenue", recipient="CFO")
        compact = env.to_compact()
        assert "@CFO" in compact
        assert "ANALYZE" in compact
        assert "revenue" in compact

    def test_to_compact_with_params(self):
        env = TypedEnvelope(
            action="analyze",
            target="revenue",
            params={"period": "Q1-2026", "compare": "Q4-2025"},
        )
        compact = env.to_compact()
        assert "Q1-2026" in compact

    def test_to_compact_with_priority(self):
        env = TypedEnvelope(action="analyze", target="revenue", priority="urgent")
        compact = env.to_compact()
        assert "!urgent" in compact

    def test_to_compact_with_modifiers(self):
        env = TypedEnvelope(
            action="analyze", target="revenue", modifiers=["discretion", "thorough"]
        )
        compact = env.to_compact()
        assert "~discretion" in compact

    def test_to_compact_with_constraints(self):
        env = TypedEnvelope(
            action="analyze", target="revenue", constraints=["within 2 hours"]
        )
        compact = env.to_compact()
        assert "within 2 hours" in compact

    def test_to_compact_with_response_format(self):
        env = TypedEnvelope(action="analyze", target="revenue", response_format="summary")
        compact = env.to_compact()
        assert "-> summary" in compact

    def test_to_compact_with_context_refs(self):
        env = TypedEnvelope(
            action="analyze", target="revenue", context_refs=["org:state@v1"]
        )
        compact = env.to_compact()
        assert "bb=[org:state@v1]" in compact

    def test_to_dict(self):
        env = TypedEnvelope(action="analyze", target="revenue", recipient="CFO")
        d = env.to_dict()
        assert d["action"] == "analyze"
        assert d["target"] == "revenue"
        assert d["recipient"] == "CFO"

    def test_to_dict_omits_empty(self):
        env = TypedEnvelope(action="analyze", target="revenue")
        d = env.to_dict()
        assert "params" not in d
        assert "constraints" not in d
        assert "sender" not in d

    def test_from_dict(self):
        d = {"action": "analyze", "target": "revenue", "recipient": "CFO"}
        env = TypedEnvelope.from_dict(d)
        assert env.action == "analyze"
        assert env.recipient == "CFO"

    def test_roundtrip_dict(self):
        original = TypedEnvelope(
            action="analyze",
            target="revenue",
            params={"period": "Q1-2026"},
            recipient="CFO",
            priority="high",
            modifiers=["thorough"],
        )
        restored = TypedEnvelope.from_dict(original.to_dict())
        assert restored.action == original.action
        assert restored.target == original.target
        assert restored.params == original.params
        assert restored.recipient == original.recipient
        assert restored.priority == original.priority

    def test_to_json(self):
        env = TypedEnvelope(action="analyze", target="revenue")
        j = env.to_json()
        assert '"action":"analyze"' in j

    def test_normal_priority_not_in_compact(self):
        env = TypedEnvelope(action="analyze", target="revenue", priority="normal")
        compact = env.to_compact()
        assert "!normal" not in compact
