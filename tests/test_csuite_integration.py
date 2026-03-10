"""Tests for C-Suite integration adapter."""

from prompt_optimizer.integrations.csuite import CSuitePromptOptimizer
from tests.conftest import CONTEXT_HEAVY_PROMPT, POLITE_PROMPT


class TestCSuitePromptOptimizer:
    def setup_method(self):
        self.adapter = CSuitePromptOptimizer()

    def test_optimize_message(self):
        result = self.adapter.optimize_message(
            payload=POLITE_PROMPT,
            sender="CEO",
            recipient="CFO",
        )
        assert result.compressed_text
        assert result.token_counts.compressed <= result.token_counts.original

    def test_optimize_task_prompt(self):
        result = self.adapter.optimize_task_prompt(
            task_description=POLITE_PROMPT,
            agent_code="CFO",
        )
        assert result.compressed_text
        assert len(result.layers_applied) > 0

    def test_optimize_with_org_context(self):
        result = self.adapter.optimize_task_prompt(
            task_description=CONTEXT_HEAVY_PROMPT,
            agent_code="CFO",
            org_context={"revenue": "$2.3M", "growth": "12.4%"},
        )
        assert result.compressed_text

    def test_optimize_with_history(self):
        result = self.adapter.optimize_message(
            payload="As mentioned earlier, analyze the trends.",
            history=["Revenue was $2.3M in Q1 2026."],
        )
        assert result.compressed_text

    def test_restore(self):
        result = self.adapter.optimize_message(payload=POLITE_PROMPT)
        restored = self.adapter.restore(result)
        assert restored
        assert len(restored) > 0

    def test_default_max_layer(self):
        assert self.adapter.default_max_layer == 2

    def test_custom_max_layer(self):
        adapter = CSuitePromptOptimizer(default_max_layer=1)
        result = adapter.optimize_message(payload=POLITE_PROMPT)
        assert all(l <= 1 for l in result.layers_applied)

    def test_store_org_context(self):
        pointer = self.adapter.store_org_context("revenue", "$2.3M")
        assert "org:revenue" in pointer

    def test_store_agent_context(self):
        pointer = self.adapter.store_agent_context("CTO", "status", "active")
        assert "agent:CTO" in pointer

    def test_executive_codes_complete(self):
        assert len(CSuitePromptOptimizer.EXECUTIVE_CODES) == 16
        assert "CEO" in CSuitePromptOptimizer.EXECUTIVE_CODES
        assert "CSUSO" in CSuitePromptOptimizer.EXECUTIVE_CODES
