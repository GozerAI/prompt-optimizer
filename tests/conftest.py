"""Shared test fixtures for prompt-optimizer."""

import pytest

from prompt_optimizer.blackboard import Blackboard
from prompt_optimizer.schema_registry import SchemaRegistry
from prompt_optimizer.types import CompressionContext


@pytest.fixture
def blackboard():
    return Blackboard()


@pytest.fixture
def schema_registry():
    return SchemaRegistry()


@pytest.fixture
def context(blackboard, schema_registry):
    return CompressionContext(
        agent_codes=["CEO", "COO", "CTO", "CFO", "CIO", "CMO"],
        blackboard=blackboard,
        schema_registry=schema_registry,
    )


@pytest.fixture
def context_with_history(context):
    context.history = [
        "The Q1 2026 revenue was $2.3M with a growth rate of 12.4%.",
        "The board mandated maintaining 20% profit margins.",
        "CTO assessed infrastructure costs will increase by 15%.",
    ]
    return context


# Sample prompts of varying complexity
SIMPLE_PROMPT = "Please analyze the Q1 2026 revenue data."

POLITE_PROMPT = (
    "Hey CFO, could you please take a careful look at the Q1 2026 revenue "
    "numbers and give me a detailed breakdown of the growth rate compared "
    "to last quarter, along with any key metrics you think are relevant."
)

MULTI_STEP_PROMPT = (
    "First, have the CTO assess the technical risk of the new deployment. "
    "Then, the CFO should estimate the cost impact based on the CTO's risk assessment. "
    "Finally, the CEO needs to make a go/no-go decision based on both reports."
)

CONTEXT_HEAVY_PROMPT = (
    "Given our company revenue of $2.3M for Q1 2026, up 12.4% from Q4, "
    "the board's mandate to maintain 20% margins, the pending Series B "
    "due diligence, and the CTO's assessment that infrastructure costs "
    "will increase 15% if we proceed with the microservices migration, "
    "please analyze whether we should proceed with the migration."
)

MINIMAL_PROMPT = "Analyze revenue."


@pytest.fixture
def simple_prompt():
    return SIMPLE_PROMPT


@pytest.fixture
def polite_prompt():
    return POLITE_PROMPT


@pytest.fixture
def multi_step_prompt():
    return MULTI_STEP_PROMPT


@pytest.fixture
def context_heavy_prompt():
    return CONTEXT_HEAVY_PROMPT
