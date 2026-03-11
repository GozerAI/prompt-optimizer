"""Renderer — converts AST nodes back to compact wire format.

Round-trip property: render(parse(text)) == text for well-formed inputs.
"""

from __future__ import annotations

from prompt_optimizer.grammar.ast_nodes import (
    ASTNode,
    ConditionalNode,
    DirectiveNode,
    PipelineNode,
)


class Renderer:
    """Renders AST nodes to compact agent communication notation."""

    def render(self, node: ASTNode) -> str:
        """Render any AST node to compact text."""
        if isinstance(node, ConditionalNode):
            return self._render_conditional(node)
        if isinstance(node, PipelineNode):
            return self._render_pipeline(node)
        if isinstance(node, DirectiveNode):
            return self._render_directive(node)
        return str(node)

    def render_human(self, node: ASTNode) -> str:
        """Render AST to human-readable natural language."""
        if isinstance(node, ConditionalNode):
            return self._render_conditional_human(node)
        if isinstance(node, PipelineNode):
            return self._render_pipeline_human(node)
        if isinstance(node, DirectiveNode):
            return self._render_directive_human(node)
        return str(node)

    # --- Compact rendering ---

    def _render_directive(self, node: DirectiveNode) -> str:
        parts: list[str] = []

        if node.recipient:
            parts.append(f"@{node.recipient.agent_code}")

        parts.append(node.action)
        parts.append(node.target)

        if node.params and node.params.params:
            param_strs = []
            for p in node.params.params:
                if p.value is not None:
                    param_strs.append(f"{p.key}={p.value}")
                else:
                    param_strs.append(p.key)
            parts.append("{" + ", ".join(param_strs) + "}")

        if node.constraints and node.constraints.constraints:
            constraint_strs = [c.text for c in node.constraints.constraints]
            parts.append("[" + ", ".join(constraint_strs) + "]")

        if node.output:
            parts.append(f"-> {node.output.format}")

        if node.priority:
            parts.append(f"!{node.priority.level}")

        for mod in node.modifiers:
            parts.append(f"~{mod.name}")

        return " ".join(parts)

    def _render_pipeline(self, node: PipelineNode) -> str:
        return " | ".join(self._render_directive(d) for d in node.directives)

    def _render_conditional(self, node: ConditionalNode) -> str:
        cond = node.condition
        if cond.comparator == ":" and cond.right:
            cond_str = f"{cond.left.field}:{cond.right.field}"
        elif cond.right:
            cond_str = f"{cond.left.field} {cond.comparator} {cond.right.field}"
        else:
            cond_str = cond.left.field

        result = f"IF {cond_str} THEN {self.render(node.then_branch)}"
        if node.else_branch:
            result += f" ELSE {self.render(node.else_branch)}"
        return result

    # --- Human-readable rendering ---

    def _render_directive_human(self, node: DirectiveNode) -> str:
        parts: list[str] = []

        if node.recipient:
            parts.append(f"To {node.recipient.agent_code}:")

        parts.append(f"{node.action.lower()} {node.target}")

        if node.params and node.params.params:
            for p in node.params.params:
                if p.value is not None:
                    parts.append(f"({p.key}: {p.value})")

        if node.constraints and node.constraints.constraints:
            for c in node.constraints.constraints:
                parts.append(f"(constraint: {c.text})")

        if node.output:
            parts.append(f"and return as {node.output.format}")

        if node.priority:
            parts.append(f"[{node.priority.level} priority]")

        return " ".join(parts)

    def _render_pipeline_human(self, node: PipelineNode) -> str:
        ordinals = ["First", "Then", "Finally"]
        parts: list[str] = []
        for i, directive in enumerate(node.directives):
            prefix = ordinals[i] if i < len(ordinals) else f"Step {i + 1}"
            parts.append(f"{prefix}, {self._render_directive_human(directive)}.")
        return " ".join(parts)

    def _render_conditional_human(self, node: ConditionalNode) -> str:
        cond = node.condition
        if cond.comparator == ":" and cond.right:
            cond_str = f"{cond.left.field} is {cond.right.field}"
        elif cond.right:
            cond_str = f"{cond.left.field} {cond.comparator} {cond.right.field}"
        else:
            cond_str = cond.left.field

        result = f"If {cond_str}, then {self.render_human(node.then_branch)}"
        if node.else_branch:
            result += f" Otherwise, {self.render_human(node.else_branch)}"
        return result
