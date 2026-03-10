"""Tests for Blackboard state store."""

import pytest

from prompt_optimizer.blackboard import Blackboard


class TestBlackboard:
    def setup_method(self):
        self.bb = Blackboard()

    def test_put_and_get(self):
        pointer = self.bb.put("org", "state", {"revenue": "$2.3M"})
        assert "org:state@v1" == pointer
        assert self.bb.get(pointer) == {"revenue": "$2.3M"}

    def test_versioning(self):
        p1 = self.bb.put("org", "state", "v1_data")
        p2 = self.bb.put("org", "state", "v2_data")
        assert "v1" in p1
        assert "v2" in p2
        assert self.bb.get(p1) == "v1_data"
        assert self.bb.get(p2) == "v2_data"

    def test_no_version_bump_if_unchanged(self):
        p1 = self.bb.put("org", "state", "same_data")
        p2 = self.bb.put("org", "state", "same_data")
        assert p1 == p2  # Same pointer, no new version

    def test_get_latest(self):
        self.bb.put("org", "state", "v1")
        self.bb.put("org", "state", "v2")
        # Get without version should return latest
        assert self.bb.get("org:state") == "v2"

    def test_has(self):
        self.bb.put("org", "state", "data")
        assert self.bb.has("org:state@v1")
        assert self.bb.has("org:state")
        assert not self.bb.has("org:missing@v1")

    def test_get_missing_raises(self):
        with pytest.raises(KeyError):
            self.bb.get("org:missing")

    def test_get_wrong_version_raises(self):
        self.bb.put("org", "state", "data")
        with pytest.raises(KeyError):
            self.bb.get("org:state@v99")

    def test_snapshot(self):
        self.bb.put("org", "state", "data1")
        self.bb.put("financial", "revenue", "$2.3M")
        snapshot = self.bb.snapshot()
        assert len(snapshot) == 2

    def test_diff(self):
        other = Blackboard()
        self.bb.put("org", "state", "data1")
        other.put("org", "state", "data2")
        other.put("org", "extra", "data3")

        divergent = self.bb.diff(other)
        assert "org:state" in divergent
        assert "org:extra" in divergent

    def test_diff_no_differences(self):
        other = Blackboard()
        self.bb.put("org", "state", "same")
        other.put("org", "state", "same")
        assert self.bb.diff(other) == []

    def test_clear(self):
        self.bb.put("org", "state", "data")
        self.bb.clear()
        assert not self.bb.has("org:state")

    def test_get_latest_pointer(self):
        self.bb.put("org", "state", "v1")
        self.bb.put("org", "state", "v2")
        pointer = self.bb.get_latest_pointer("org", "state")
        assert pointer == "org:state@v2"

    def test_get_latest_pointer_missing(self):
        assert self.bb.get_latest_pointer("org", "missing") is None

    def test_multiple_namespaces(self):
        self.bb.put("org", "state", "org_data")
        self.bb.put("agent:CTO", "status", "active")
        assert self.bb.get("org:state") == "org_data"
        assert self.bb.get("agent:CTO:status") == "active"
