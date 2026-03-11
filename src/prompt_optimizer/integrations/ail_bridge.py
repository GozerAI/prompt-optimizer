"""Bridge between prompt-optimizer TypedEnvelope/grammar and AIL.

Converts prompt-optimizer DirectiveNode/PipelineNode ASTs to AIL ASTs
and vice versa.  Also provides compact_via_ail() for using AIL as the
L1 wire format when the ail package is available.

All AIL imports are lazy — the bridge degrades gracefully when ail is
not installed.
"""

from __future__ import annotations

from typing import Any


def ail_available() -> bool:
    """Return True if the ail package is importable."""
    try:
        import ail  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Grammar DirectiveNode → AIL Directive
# ---------------------------------------------------------------------------

def directive_to_ail(node: Any) -> Any:
    """Convert a prompt-optimizer DirectiveNode to an AIL Directive.

    Parameters
    ----------
    node : DirectiveNode | PipelineNode
        A grammar AST node from the prompt-optimizer.

    Returns
    -------
    ail.ast.nodes.Directive | ail.ast.nodes.Pipeline
    """
    from ail.ast.nodes import Directive, Pipeline, ResponseContract

    from prompt_optimizer.grammar.ast_nodes import DirectiveNode, PipelineNode

    if isinstance(node, PipelineNode):
        steps = [directive_to_ail(d) for d in node.directives]
        return Pipeline(steps=steps) if len(steps) > 1 else steps[0]

    if not isinstance(node, DirectiveNode):
        raise TypeError(f"Expected DirectiveNode or PipelineNode, got {type(node).__name__}")

    agent = node.recipient.agent_code if node.recipient else ""
    action = node.action.upper()
    target = node.target or ""

    params: dict[str, str] = {}
    if node.params:
        for p in node.params.params:
            if p.value is not None:
                params[p.key] = p.value
            else:
                params[p.key] = "true"  # bare keyword like 'inherit'

    constraints: list[str] = []
    if node.constraints:
        constraints = [c.text for c in node.constraints.constraints]

    response: ResponseContract | None = None
    if node.output:
        response = ResponseContract(format_hint=node.output.format)

    priority: str | None = None
    if node.priority:
        priority = node.priority.level

    modifiers: list[str] = [m.name for m in node.modifiers]

    return Directive(
        agent=agent,
        action=action,
        target=target,
        params=params,
        constraints=constraints,
        context_refs=[],
        response=response,
        priority=priority,
        modifiers=modifiers,
        retry=None,
    )


# ---------------------------------------------------------------------------
# AIL Directive → Grammar DirectiveNode
# ---------------------------------------------------------------------------

def ail_to_directive(ail_node: Any) -> Any:
    """Convert an AIL Directive to a prompt-optimizer DirectiveNode.

    Parameters
    ----------
    ail_node : ail.ast.nodes.Directive | ail.ast.nodes.Pipeline
        An AIL AST node.

    Returns
    -------
    DirectiveNode | PipelineNode
    """
    from ail.ast.nodes import Directive, Pipeline

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

    if isinstance(ail_node, Pipeline):
        directives = [ail_to_directive(s) for s in ail_node.steps]
        return PipelineNode(directives=directives)

    if not isinstance(ail_node, Directive):
        raise TypeError(f"Expected AIL Directive or Pipeline, got {type(ail_node).__name__}")

    recipient = RecipientNode(agent_code=ail_node.agent) if ail_node.agent else None

    params_node = None
    if ail_node.params:
        params_node = ParamsNode(
            params=[ParamNode(key=k, value=v) for k, v in ail_node.params.items()]
        )

    constraints_node = None
    if ail_node.constraints:
        constraints_node = ConstraintsNode(
            constraints=[ConstraintNode(text=c) for c in ail_node.constraints]
        )

    output_node = None
    if ail_node.response:
        hint = ail_node.response.format_hint or ""
        if not hint and ail_node.response.fields:
            hint = "contract"
        if hint:
            output_node = OutputNode(format=hint)

    priority_node = None
    if ail_node.priority:
        priority_node = PriorityNode(level=ail_node.priority)

    modifiers = [ModifierNode(name=m) for m in (ail_node.modifiers or [])]

    return DirectiveNode(
        action=ail_node.action.lower(),
        target=ail_node.target if isinstance(ail_node.target, str) else "",
        recipient=recipient,
        params=params_node,
        constraints=constraints_node,
        output=output_node,
        priority=priority_node,
        modifiers=modifiers,
    )


# ---------------------------------------------------------------------------
# Convenience: compress text to AIL wire format
# ---------------------------------------------------------------------------

def compact_via_ail(text: str) -> str | None:
    """Try to compile natural language to AIL wire format.

    Returns the AIL string on success, or None if AIL is unavailable
    or compilation fails.
    """
    try:
        from ail import compile_nl, emit
        ast = compile_nl(text)
        if ast is not None:
            return emit(ast)
    except Exception:
        pass
    return None


def parse_compact_via_ail(ail_text: str) -> Any | None:
    """Parse an AIL string back to an AIL AST node.

    Returns the AST on success, None on failure.
    """
    try:
        from ail import parse
        return parse(ail_text)
    except Exception:
        return None
