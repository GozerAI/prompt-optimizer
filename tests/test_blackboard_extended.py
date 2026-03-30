"""Tests for Blackboard upgrades: subscriptions, staleness, restore."""

import time
from unittest.mock import MagicMock


from prompt_optimizer.blackboard import Blackboard


class TestBlackboardSubscriptions:
    def setup_method(self):
        self.bb = Blackboard()

    def test_subscribe_notified_on_put(self):
        callback = MagicMock()
        self.bb.subscribe("org", callback)
        self.bb.put("org", "state", "data")
        callback.assert_called_once_with("org", "state", "data")

    def test_subscribe_not_notified_for_other_namespace(self):
        callback = MagicMock()
        self.bb.subscribe("org", callback)
        self.bb.put("financial", "revenue", "$2.3M")
        callback.assert_not_called()

    def test_multiple_subscribers(self):
        cb1 = MagicMock()
        cb2 = MagicMock()
        self.bb.subscribe("org", cb1)
        self.bb.subscribe("org", cb2)
        self.bb.put("org", "state", "data")
        cb1.assert_called_once()
        cb2.assert_called_once()

    def test_unsubscribe(self):
        callback = MagicMock()
        self.bb.subscribe("org", callback)
        self.bb.unsubscribe("org", callback)
        self.bb.put("org", "state", "data")
        callback.assert_not_called()

    def test_subscriber_error_doesnt_break_put(self):
        def bad_callback(ns, key, val):
            raise RuntimeError("boom")

        self.bb.subscribe("org", bad_callback)
        # Should not raise
        pointer = self.bb.put("org", "state", "data")
        assert self.bb.has(pointer)

    def test_no_notification_on_unchanged_value(self):
        callback = MagicMock()
        self.bb.subscribe("org", callback)
        self.bb.put("org", "state", "data")
        self.bb.put("org", "state", "data")  # Same value
        assert callback.call_count == 1  # Only called once


class TestBlackboardStaleness:
    def setup_method(self):
        self.bb = Blackboard(staleness_threshold=1.0)

    def test_fresh_entry_not_stale(self):
        self.bb.put("org", "state", "data")
        assert self.bb.get_stale(threshold=1.0) == []

    def test_old_entry_is_stale(self):
        self.bb.put("org", "state", "data")
        # Manually age the entry
        store_key = "org:state"
        self.bb._store[store_key][-1].timestamp = time.time() - 2.0
        stale = self.bb.get_stale(threshold=1.0)
        assert len(stale) == 1

    def test_is_stale_method(self):
        pointer = self.bb.put("org", "state", "data")
        assert not self.bb.is_stale(pointer, threshold=1.0)
        self.bb._store["org:state"][-1].timestamp = time.time() - 2.0
        assert self.bb.is_stale(pointer, threshold=1.0)

    def test_missing_pointer_is_stale(self):
        assert self.bb.is_stale("org:missing@v1")

    def test_default_threshold_used(self):
        bb = Blackboard(staleness_threshold=0.001)
        bb.put("org", "state", "data")
        time.sleep(0.01)
        stale = bb.get_stale()
        assert len(stale) == 1


class TestBlackboardNamespaces:
    def test_namespaces_empty(self):
        bb = Blackboard()
        assert bb.namespaces == []

    def test_namespaces_populated(self):
        bb = Blackboard()
        bb.put("org", "state", "data")
        bb.put("financial", "revenue", "$2.3M")
        bb.put("org", "goals", "growth")
        assert bb.namespaces == ["financial", "org"]


class TestBlackboardRestore:
    def test_snapshot_restore_roundtrip(self):
        bb1 = Blackboard()
        bb1.put("org", "state", "data1")
        bb1.put("financial", "revenue", "$2.3M")
        snapshot = bb1.snapshot()

        bb2 = Blackboard()
        bb2.restore(snapshot)
        assert bb2.get("org:state") == "data1"
        assert bb2.get("financial:revenue") == "$2.3M"

    def test_restore_clears_existing(self):
        bb = Blackboard()
        bb.put("old", "key", "value")
        bb.restore({"org:state@v1": "new_data"})
        assert not bb.has("old:key")
        assert bb.get("org:state") == "new_data"


class TestBlackboardSourceAgent:
    def test_source_agent_stored(self):
        bb = Blackboard()
        bb.put("org", "state", "data", source_agent="CFO")
        entry = bb._store["org:state"][-1]
        assert entry.source_agent == "CFO"

    def test_source_agent_none_by_default(self):
        bb = Blackboard()
        bb.put("org", "state", "data")
        entry = bb._store["org:state"][-1]
        assert entry.source_agent is None
