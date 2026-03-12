"""Validate AST nodes for semantic correctness."""

from __future__ import annotations

from dataclasses import dataclass, field

from prompt_optimizer.grammar.ast_nodes import (
    ASTNode,
    ConditionalNode,
    DirectiveNode,
    ParallelBlockNode,
    PipelineNode,
    PrevRefNode,
    ProgramNode,
    SequentialBlockNode,
)


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0


class Validator:
    """Validate AST for semantic issues."""

    def __init__(self, known_agents: set[str] | None = None) -> None:
        self._known_agents = known_agents or set()

    def validate(self, node: ASTNode) -> ValidationResult:
        result = ValidationResult()
        self._check(node, result, in_pipeline=False, step_index=0)
        return result

    def _check(self, node: ASTNode, result: ValidationResult, *, in_pipeline: bool, step_index: int) -> None:
        if isinstance(node, ProgramNode):
            for i, stmt in enumerate(node.statements):
                self._check(stmt, result, in_pipeline=False, step_index=i)

        elif isinstance(node, DirectiveNode):
            self._check_directive(node, result, in_pipeline=in_pipeline, step_index=step_index)

        elif isinstance(node, PipelineNode):
            for i, step in enumerate(node.directives):
                self._check(step, result, in_pipeline=True, step_index=i)

        elif isinstance(node, ParallelBlockNode):
            for branch in node.branches:
                self._check(branch, result, in_pipeline=False, step_index=step_index)
            # Warn about $prev in parallel branches
            for branch in node.branches:
                if self._has_prev_ref(branch):
                    result.warnings.append(
                        "$prev reference in PAR block is ambiguous — parallel branches have no ordering"
                    )
                    break

        elif isinstance(node, SequentialBlockNode):
            for i, step in enumerate(node.steps):
                self._check(step, result, in_pipeline=True, step_index=i)

        elif isinstance(node, ConditionalNode):
            condition = node.condition
            if isinstance(condition, str):
                if not condition.strip():
                    result.errors.append("IF condition is empty")
            else:
                # ConditionNode — check left field exists
                pass
            self._check(node.then_branch, result, in_pipeline=False, step_index=step_index)
            if node.else_branch:
                self._check(node.else_branch, result, in_pipeline=False, step_index=step_index)

    def _check_directive(self, d: DirectiveNode, result: ValidationResult, *, in_pipeline: bool, step_index: int) -> None:
        agent = d.recipient.agent_code if d.recipient else ""
        if not agent:
            result.errors.append("Directive missing agent")
        if not d.action:
            result.errors.append("Directive missing action")
        if self._known_agents and agent and agent not in self._known_agents:
            result.warnings.append(f"Unknown agent: @{agent}")
        if isinstance(d.target, PrevRefNode) and not in_pipeline and step_index == 0:
            result.errors.append(
                f"$prev used in @{agent} {d.action} but no prior step exists"
            )
        # Check params for $prev references at step 0
        if d.params:
            for p in d.params.params:
                if p.value and isinstance(p.value, str) and p.value.startswith("$prev"):
                    if not in_pipeline and step_index == 0:
                        result.errors.append(
                            f"$prev in params of @{agent} {d.action} but no prior step exists"
                        )

    def _has_prev_ref(self, node: ASTNode) -> bool:
        if isinstance(node, DirectiveNode):
            if isinstance(node.target, PrevRefNode):
                return True
            return False
        if isinstance(node, PipelineNode):
            return any(self._has_prev_ref(s) for s in node.directives)
        if isinstance(node, ParallelBlockNode):
            return any(self._has_prev_ref(b) for b in node.branches)
        if isinstance(node, SequentialBlockNode):
            return any(self._has_prev_ref(s) for s in node.steps)
        return False
