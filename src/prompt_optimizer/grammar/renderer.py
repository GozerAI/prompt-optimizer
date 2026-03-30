"""Renderer — converts AST nodes back to compact wire format.

Round-trip property: render(parse(text)) == text for well-formed inputs.
"""

from __future__ import annotations

from prompt_optimizer.grammar.ast_nodes import (
    ASTNode,
    AgentFieldRefNode,
    BlackboardRefNode,
    ConditionalNode,
    ContractFieldNode,
    DirectiveNode,
    ParallelBlockNode,
    PipelineNode,
    PrevRefNode,
    ProgramNode,
    ResponseContractNode,
    RetryPolicyNode,
    SequentialBlockNode,
)


class Renderer:
    """Renders AST nodes to compact agent communication notation."""

    def render(self, node: ASTNode) -> str:
        """Render any AST node to compact text."""
        if isinstance(node, ProgramNode):
            return self._render_program(node)
        if isinstance(node, ConditionalNode):
            return self._render_conditional(node)
        if isinstance(node, PipelineNode):
            return self._render_pipeline(node)
        if isinstance(node, ParallelBlockNode):
            return self._render_parallel(node)
        if isinstance(node, SequentialBlockNode):
            return self._render_sequential(node)
        if isinstance(node, DirectiveNode):
            return self._render_directive(node)
        return str(node)

    def render_human(self, node: ASTNode) -> str:
        """Render AST to human-readable natural language."""
        if isinstance(node, ProgramNode):
            return " ".join(self.render_human(s) for s in node.statements)
        if isinstance(node, ConditionalNode):
            return self._render_conditional_human(node)
        if isinstance(node, PipelineNode):
            return self._render_pipeline_human(node)
        if isinstance(node, ParallelBlockNode):
            parts = [self.render_human(b) for b in node.branches]
            return "Simultaneously: " + "; ".join(parts)
        if isinstance(node, SequentialBlockNode):
            parts = [self.render_human(s) for s in node.steps]
            return " Then, ".join(parts)
        if isinstance(node, DirectiveNode):
            return self._render_directive_human(node)
        return str(node)

    # --- Compact rendering ---

    def _render_program(self, node: ProgramNode) -> str:
        return "\n".join(self.render(s) for s in node.statements)

    def _render_directive(self, node: DirectiveNode) -> str:
        parts: list[str] = []

        if node.recipient:
            parts.append(f"@{node.recipient.agent_code}")

        parts.append(node.action)

        # Target
        if node.target:
            parts.append(self._render_target(node.target))

        # Params
        if node.params and node.params.params:
            param_strs = []
            for p in node.params.params:
                if p.value is not None:
                    param_strs.append(f"{p.key}={p.value}")
                else:
                    param_strs.append(p.key)
            parts.append("{" + ", ".join(param_strs) + "}")

        # Constraints
        if node.constraints and node.constraints.constraints:
            constraint_strs = [c.text for c in node.constraints.constraints]
            parts.append("[" + ", ".join(constraint_strs) + "]")

        # Blackboard refs
        if node.context_refs:
            refs = ", ".join(f"bb:{r.pointer}" for r in node.context_refs)
            parts.append(f"[{refs}]")

        # Response contract (takes precedence over simple output)
        if node.response_contract:
            parts.append(self._render_response_contract(node.response_contract))
        elif node.output:
            parts.append(f"-> {node.output.format}")

        # Priority
        if node.priority:
            parts.append(f"!{node.priority.level}")

        # Modifiers
        for mod in node.modifiers:
            parts.append(f"~{mod.name}")

        # Retry
        if node.retry:
            parts.append(self._render_retry(node.retry))

        return " ".join(parts)

    def _render_pipeline(self, node: PipelineNode) -> str:
        return " | ".join(self._render_directive(d) for d in node.directives)

    def _render_parallel(self, node: ParallelBlockNode) -> str:
        body = "; ".join(self.render(b) for b in node.branches)
        return f"PAR {{ {body} }}"

    def _render_sequential(self, node: SequentialBlockNode) -> str:
        body = "; ".join(self.render(s) for s in node.steps)
        return f"SEQ {{ {body} }}"

    def _render_conditional(self, node: ConditionalNode) -> str:
        if isinstance(node.condition, str):
            cond_str = node.condition
        else:
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

    # --- Helpers ---

    def _render_target(self, target) -> str:
        if isinstance(target, PrevRefNode):
            return "$prev" if target.step is None else f"$prev[{target.step}]"
        if isinstance(target, BlackboardRefNode):
            return f"bb:{target.pointer}"
        if isinstance(target, AgentFieldRefNode):
            if target.field:
                return f"@{target.agent}.{target.field}"
            return f"@{target.agent}"
        return str(target)

    def _render_response_contract(self, contract: ResponseContractNode) -> str:
        if contract.fields:
            fields = ", ".join(self._render_contract_field(f) for f in contract.fields)
            return f"-> {{{fields}}}"
        if contract.format_hint:
            return f"-> {contract.format_hint}"
        return ""

    def _render_contract_field(self, f: ContractFieldNode) -> str:
        opt = "?" if not f.required else ""
        return f"{f.name}{opt}: {f.type_hint}"

    def _render_retry(self, retry: RetryPolicyNode) -> str:
        parts = [f"RETRY {retry.max_retries}"]
        if retry.backoff != "exp":
            parts.append(f"BACKOFF {retry.backoff}")
        if retry.fallback_agent:
            parts.append(f"FALLBACK @{retry.fallback_agent}")
        return " ".join(parts)

    # --- Human-readable rendering ---

    def _render_directive_human(self, node: DirectiveNode) -> str:
        parts: list[str] = []

        if node.recipient:
            parts.append(f"To {node.recipient.agent_code}:")

        parts.append(f"{node.action.lower()} {self._render_target(node.target) if not isinstance(node.target, str) else node.target}")

        if node.params and node.params.params:
            for p in node.params.params:
                if p.value is not None:
                    parts.append(f"({p.key}: {p.value})")

        if node.constraints and node.constraints.constraints:
            for c in node.constraints.constraints:
                parts.append(f"(constraint: {c.text})")

        if node.response_contract and node.response_contract.format_hint:
            parts.append(f"and return as {node.response_contract.format_hint}")
        elif node.output:
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
        if isinstance(node.condition, str):
            cond_str = node.condition
        else:
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
