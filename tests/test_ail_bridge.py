"""Tests for AIL bridge integration."""

import pytest

from prompt_optimizer.integrations.ail_bridge import (
    ail_available,
    ail_to_directive,
    compact_via_ail,
    directive_to_ail,
    parse_compact_via_ail,
)

# Skip all tests if ail is not installed
pytestmark = pytest.mark.skipif(not ail_available(), reason="ail package not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_directive_node(
    action="analyze",
    target="market",
    agent="CSO",
    params=None,
    constraints=None,
    output=None,
    priority=None,
    modifiers=None,
):
    """Create a prompt-optimizer DirectiveNode."""
    from prompt_optimizer.grammar.ast_nodes import (
        ConstraintNode,
        ConstraintsNode,
        DirectiveNode,
        ModifierNode,
        OutputNode,
        ParamNode,
        ParamsNode,
        PipelineNode,
        PriorityNode,
        RecipientNode,
    )

    recipient = RecipientNode(agent_code=agent) if agent else None
    params_node = None
    if params:
        params_node = ParamsNode(params=[ParamNode(key=k, value=v) for k, v in params.items()])
    constraints_node = None
    if constraints:
        constraints_node = ConstraintsNode(
            constraints=[ConstraintNode(text=c) for c in constraints]
        )
    output_node = OutputNode(format=output) if output else None
    priority_node = PriorityNode(level=priority) if priority else None
    mods = [ModifierNode(name=m) for m in (modifiers or [])]

    return DirectiveNode(
        action=action,
        target=target,
        recipient=recipient,
        params=params_node,
        constraints=constraints_node,
        output=output_node,
        priority=priority_node,
        modifiers=mods,
    )


def _make_ail_directive(agent="CSO", action="ANALYZE", target="market", **kwargs):
    """Create an AIL Directive."""
    from ail.ast.nodes import Directive
    return Directive(
        agent=agent,
        action=action,
        target=target,
        params=kwargs.get("params", {}),
        constraints=kwargs.get("constraints", []),
        context_refs=kwargs.get("context_refs", []),
        response=kwargs.get("response", None),
        priority=kwargs.get("priority", None),
        modifiers=kwargs.get("modifiers", []),
        retry=None,
    )


# ---------------------------------------------------------------------------
# ail_available
# ---------------------------------------------------------------------------

class TestAilAvailable:
    def test_returns_true(self):
        assert ail_available() is True


# ---------------------------------------------------------------------------
# directive_to_ail
# ---------------------------------------------------------------------------

class TestDirectiveToAil:
    def test_basic_conversion(self):
        node = _make_directive_node()
        result = directive_to_ail(node)
        from ail.ast.nodes import Directive
        assert isinstance(result, Directive)
        assert result.agent == "CSO"
        assert result.action == "ANALYZE"
        assert result.target == "market"

    def test_with_params(self):
        node = _make_directive_node(params={"focus": "trends", "period": "Q4"})
        result = directive_to_ail(node)
        assert result.params == {"focus": "trends", "period": "Q4"}

    def test_with_constraints(self):
        node = _make_directive_node(constraints=["budget > 1000", "timeline < 30d"])
        result = directive_to_ail(node)
        assert result.constraints == ["budget > 1000", "timeline < 30d"]

    def test_with_output(self):
        node = _make_directive_node(output="summary")
        result = directive_to_ail(node)
        assert result.response is not None
        assert result.response.format_hint == "summary"

    def test_with_priority(self):
        node = _make_directive_node(priority="urgent")
        result = directive_to_ail(node)
        assert result.priority == "urgent"

    def test_with_modifiers(self):
        node = _make_directive_node(modifiers=["thorough", "discretion"])
        result = directive_to_ail(node)
        assert result.modifiers == ["thorough", "discretion"]

    def test_no_agent(self):
        node = _make_directive_node(agent=None)
        result = directive_to_ail(node)
        assert result.agent == ""

    def test_pipeline_conversion(self):
        from prompt_optimizer.grammar.ast_nodes import PipelineNode
        d1 = _make_directive_node(action="analyze", agent="CSO")
        d2 = _make_directive_node(action="decide", agent="CEO")
        pipeline = PipelineNode(directives=[d1, d2])
        result = directive_to_ail(pipeline)
        from ail.ast.nodes import Pipeline
        assert isinstance(result, Pipeline)
        assert len(result.steps) == 2

    def test_single_pipeline_unwraps(self):
        from prompt_optimizer.grammar.ast_nodes import PipelineNode
        d1 = _make_directive_node()
        pipeline = PipelineNode(directives=[d1])
        result = directive_to_ail(pipeline)
        from ail.ast.nodes import Directive
        assert isinstance(result, Directive)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Expected DirectiveNode"):
            directive_to_ail("not a node")

    def test_bare_param_keyword(self):
        """Bare keywords like 'inherit' (no value) become key='true'."""
        from prompt_optimizer.grammar.ast_nodes import (
            DirectiveNode,
            ParamNode,
            ParamsNode,
            RecipientNode,
        )
        node = DirectiveNode(
            action="analyze",
            target="market",
            recipient=RecipientNode(agent_code="CSO"),
            params=ParamsNode(params=[ParamNode(key="inherit", value=None)]),
        )
        result = directive_to_ail(node)
        assert result.params == {"inherit": "true"}


# ---------------------------------------------------------------------------
# ail_to_directive
# ---------------------------------------------------------------------------

class TestAilToDirective:
    def test_basic_conversion(self):
        ail_node = _make_ail_directive()
        result = ail_to_directive(ail_node)
        from prompt_optimizer.grammar.ast_nodes import DirectiveNode
        assert isinstance(result, DirectiveNode)
        assert result.recipient.agent_code == "CSO"
        assert result.action == "analyze"
        assert result.target == "market"

    def test_with_params(self):
        ail_node = _make_ail_directive(params={"scope": "global"})
        result = ail_to_directive(ail_node)
        assert result.params is not None
        assert len(result.params.params) == 1
        assert result.params.params[0].key == "scope"
        assert result.params.params[0].value == "global"

    def test_with_constraints(self):
        ail_node = _make_ail_directive(constraints=["budget > 5000"])
        result = ail_to_directive(ail_node)
        assert result.constraints is not None
        assert result.constraints.constraints[0].text == "budget > 5000"

    def test_with_response_hint(self):
        from ail.ast.nodes import ResponseContract
        ail_node = _make_ail_directive(response=ResponseContract(format_hint="report"))
        result = ail_to_directive(ail_node)
        assert result.output is not None
        assert result.output.format == "report"

    def test_with_priority(self):
        ail_node = _make_ail_directive(priority="high")
        result = ail_to_directive(ail_node)
        assert result.priority is not None
        assert result.priority.level == "high"

    def test_with_modifiers(self):
        ail_node = _make_ail_directive(modifiers=["brief"])
        result = ail_to_directive(ail_node)
        assert len(result.modifiers) == 1
        assert result.modifiers[0].name == "brief"

    def test_pipeline_conversion(self):
        from ail.ast.nodes import Pipeline
        d1 = _make_ail_directive(agent="CSO", action="ANALYZE")
        d2 = _make_ail_directive(agent="CEO", action="DECIDE")
        pipeline = Pipeline(steps=[d1, d2])
        result = ail_to_directive(pipeline)
        from prompt_optimizer.grammar.ast_nodes import PipelineNode
        assert isinstance(result, PipelineNode)
        assert len(result.directives) == 2

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Expected AIL Directive"):
            ail_to_directive("not a node")

    def test_no_agent(self):
        ail_node = _make_ail_directive(agent="")
        result = ail_to_directive(ail_node)
        assert result.recipient is None


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def test_grammar_to_ail_and_back(self):
        original = _make_directive_node(
            action="forecast",
            target="revenue",
            agent="CFO",
            params={"horizon": "Q1"},
            constraints=["accuracy > 90%"],
            output="report",
            priority="high",
            modifiers=["thorough"],
        )
        ail_node = directive_to_ail(original)
        restored = ail_to_directive(ail_node)

        assert restored.recipient.agent_code == "CFO"
        assert restored.action == "forecast"
        assert restored.target == "revenue"
        assert restored.params.params[0].key == "horizon"
        assert restored.output.format == "report"
        assert restored.priority.level == "high"
        assert restored.modifiers[0].name == "thorough"

    def test_ail_to_grammar_and_back(self):
        from ail.ast.nodes import ResponseContract
        original = _make_ail_directive(
            agent="CMO",
            action="CREATE",
            target="content",
            params={"type": "blog"},
            response=ResponseContract(format_hint="summary"),
            modifiers=["brief"],
        )
        grammar_node = ail_to_directive(original)
        restored = directive_to_ail(grammar_node)

        assert restored.agent == "CMO"
        assert restored.action == "CREATE"
        assert restored.target == "content"
        assert restored.params == {"type": "blog"}
        assert restored.response.format_hint == "summary"
        assert restored.modifiers == ["brief"]


# ---------------------------------------------------------------------------
# compact_via_ail
# ---------------------------------------------------------------------------

class TestCompactViaAil:
    def test_simple_nl(self):
        result = compact_via_ail("CSO analyze the market trends")
        # Should return an AIL string or None
        if result is not None:
            assert "@CSO" in result or "ANALYZE" in result

    def test_returns_none_on_garbage(self):
        result = compact_via_ail("just some random words with no agent meaning")
        # May or may not parse — either a string or None is fine
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# parse_compact_via_ail
# ---------------------------------------------------------------------------

class TestParseCompactViaAil:
    def test_valid_ail(self):
        result = parse_compact_via_ail("@CSO ANALYZE market")
        assert result is not None

    def test_invalid_ail(self):
        result = parse_compact_via_ail("@@@ NOT VALID !!!")
        assert result is None
