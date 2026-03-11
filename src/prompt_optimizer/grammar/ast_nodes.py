"""AST node types for the agent communication grammar.

Grammar:
    message     := conditional | pipeline
    pipeline    := directive ('|' directive)*
    directive   := [recipient] action target [params] [constraints] [output] [priority] [modifiers]
    conditional := 'IF' condition 'THEN' message ['ELSE' message]
    recipient   := '@' AGENT_CODE
    params      := '{' param (',' param)* '}'
    param       := KEY '=' VALUE | 'inherit'
    constraints := '[' text (',' text)* ']'
    output      := '->' IDENTIFIER
    priority    := '!' IDENTIFIER
    modifiers   := ('~' IDENTIFIER)+
    condition   := IDENTIFIER ':' IDENTIFIER | IDENTIFIER COMPARATOR IDENTIFIER
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ASTNode:
    """Base AST node."""
    pass


@dataclass
class RecipientNode(ASTNode):
    agent_code: str


@dataclass
class ParamNode(ASTNode):
    key: str
    value: Optional[str] = None  # None for bare keywords like 'inherit'


@dataclass
class ParamsNode(ASTNode):
    params: list[ParamNode] = field(default_factory=list)


@dataclass
class ConstraintNode(ASTNode):
    text: str


@dataclass
class ConstraintsNode(ASTNode):
    constraints: list[ConstraintNode] = field(default_factory=list)


@dataclass
class PriorityNode(ASTNode):
    level: str  # "urgent", "high", "low"


@dataclass
class ModifierNode(ASTNode):
    name: str  # "thorough", "brief", "discretion"


@dataclass
class OutputNode(ASTNode):
    format: str  # "summary", "report", "decision"


@dataclass
class DirectiveNode(ASTNode):
    """A single agent directive: @AGENT ACTION target {params} [constraints] -> output !priority ~mods"""

    action: str
    target: str
    recipient: Optional[RecipientNode] = None
    params: Optional[ParamsNode] = None
    constraints: Optional[ConstraintsNode] = None
    output: Optional[OutputNode] = None
    priority: Optional[PriorityNode] = None
    modifiers: list[ModifierNode] = field(default_factory=list)


@dataclass
class PipelineNode(ASTNode):
    """Two or more directives chained: directive | directive | ..."""

    directives: list[DirectiveNode] = field(default_factory=list)


@dataclass
class ExpressionNode(ASTNode):
    """A value reference: field:value or standalone value."""

    field: str
    value: Optional[str] = None  # None for standalone identifiers


@dataclass
class ConditionNode(ASTNode):
    """A condition: risk:high, cost > 1000, etc."""

    left: ExpressionNode
    comparator: str  # ":", "==", ">", "<", ">=", "<=", "!="
    right: Optional[ExpressionNode] = None  # None for shorthand like risk:high


@dataclass
class ConditionalNode(ASTNode):
    """IF condition THEN message [ELSE message]"""

    condition: ConditionNode
    then_branch: ASTNode  # DirectiveNode | PipelineNode | ConditionalNode
    else_branch: Optional[ASTNode] = None
