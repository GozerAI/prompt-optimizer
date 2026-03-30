"""AST node types for the agent communication grammar.

Grammar:
    program     := statement ((NEWLINE | SEMICOLON) statement)* EOF
    statement   := conditional | par_block | seq_block | pipeline
    par_block   := PAR '{' statement (';' statement)* '}'
    seq_block   := SEQ '{' statement (';' statement)* '}'
    pipeline    := directive ('|' directive)*
    directive   := [recipient] action target [params] [constraints] [output|contract] [priority] [modifiers] [retry]
    conditional := 'IF' condition 'THEN' message ['ELSE' message]
    recipient   := '@' AGENT_CODE
    params      := '{' param (',' param)* '}'
    param       := KEY '=' VALUE | 'inherit'
    constraints := '[' text (',' text)* ']'
    output      := '->' IDENTIFIER
    contract    := '->' '{' field_def (',' field_def)* '}'
    retry       := 'RETRY' NUMBER ['BACKOFF' IDENTIFIER] ['FALLBACK' '@' AGENT_CODE]
    priority    := '!' IDENTIFIER
    modifiers   := ('~' IDENTIFIER)+
    condition   := IDENTIFIER ':' IDENTIFIER | IDENTIFIER COMPARATOR IDENTIFIER
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


class ASTNode:
    """Base AST node."""
    pass


# ---------------------------------------------------------------------------
# Leaf / value nodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrevRefNode(ASTNode):
    """Reference to the previous step's output: $prev or $prev[N]"""

    step: int | None = None  # None = immediate predecessor, int = specific step index

    def __repr__(self) -> str:
        return "$prev" if self.step is None else f"$prev[{self.step}]"


@dataclass(frozen=True)
class BlackboardRefNode(ASTNode):
    """Reference to a blackboard entry: bb:namespace:key@vN"""

    namespace: str = ""
    key: str = ""
    version: int | None = None  # None = latest

    @property
    def pointer(self) -> str:
        base = f"{self.namespace}:{self.key}"
        return f"{base}@v{self.version}" if self.version is not None else base

    def __repr__(self) -> str:
        return f"bb:{self.pointer}"


@dataclass(frozen=True)
class AgentFieldRefNode(ASTNode):
    """Reference to an agent's field: @AGENT.field"""

    agent: str = ""
    field: str | None = None

    def __repr__(self) -> str:
        if self.field:
            return f"@{self.agent}.{self.field}"
        return f"@{self.agent}"


# ---------------------------------------------------------------------------
# Response contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContractFieldNode(ASTNode):
    """A single field in a response contract."""

    name: str = ""
    type_hint: str = ""  # "str", "float", "int", "bool", "list", "dict", "any"
    required: bool = True

    def __repr__(self) -> str:
        opt = "?" if not self.required else ""
        return f"{self.name}{opt}: {self.type_hint}"


@dataclass(frozen=True)
class ResponseContractNode(ASTNode):
    """Typed response schema: -> {confidence: float, recommendation: str}"""

    fields: tuple[ContractFieldNode, ...] = ()
    format_hint: str | None = None

    def __repr__(self) -> str:
        if self.fields:
            body = ", ".join(repr(f) for f in self.fields)
            return f"-> {{{body}}}"
        if self.format_hint:
            return f"-> {self.format_hint}"
        return ""


# ---------------------------------------------------------------------------
# Retry / error handling
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetryPolicyNode(ASTNode):
    """Error handling: RETRY 3 BACKOFF exp FALLBACK @CIO"""

    max_retries: int = 3
    backoff: str = "exp"  # "exp", "linear", "fixed"
    fallback_agent: str | None = None

    def __repr__(self) -> str:
        parts = [f"RETRY {self.max_retries}"]
        if self.backoff != "exp":
            parts.append(f"BACKOFF {self.backoff}")
        if self.fallback_agent:
            parts.append(f"FALLBACK @{self.fallback_agent}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Original simple nodes (kept for NL compiler compatibility)
# ---------------------------------------------------------------------------

@dataclass
class RecipientNode(ASTNode):
    agent_code: str = ""


@dataclass
class ParamNode(ASTNode):
    key: str = ""
    value: Optional[str] = None  # None for bare keywords like 'inherit'


@dataclass
class ParamsNode(ASTNode):
    params: list[ParamNode] = field(default_factory=list)


@dataclass
class ConstraintNode(ASTNode):
    text: str = ""


@dataclass
class ConstraintsNode(ASTNode):
    constraints: list[ConstraintNode] = field(default_factory=list)


@dataclass
class PriorityNode(ASTNode):
    level: str = ""  # "urgent", "high", "low"


@dataclass
class ModifierNode(ASTNode):
    name: str = ""  # "thorough", "brief", "discretion"


@dataclass
class OutputNode(ASTNode):
    format: str = ""  # "summary", "report", "decision"


# ---------------------------------------------------------------------------
# Core directive
# ---------------------------------------------------------------------------

@dataclass
class DirectiveNode(ASTNode):
    """A single agent directive.

    Example: @CFO ANALYZE revenue {period=Q1} [margin>20%] -> report !urgent ~thorough
    """

    action: str = ""
    target: str | PrevRefNode | BlackboardRefNode | AgentFieldRefNode = ""
    recipient: Optional[RecipientNode] = None
    params: Optional[ParamsNode] = None
    constraints: Optional[ConstraintsNode] = None
    output: Optional[OutputNode] = None
    priority: Optional[PriorityNode] = None
    modifiers: list[ModifierNode] = field(default_factory=list)
    # New fields for orchestration
    retry: Optional[RetryPolicyNode] = None
    response_contract: Optional[ResponseContractNode] = None
    context_refs: list[BlackboardRefNode] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Composition nodes
# ---------------------------------------------------------------------------

@dataclass
class PipelineNode(ASTNode):
    """Two or more directives chained: directive | directive | ..."""

    directives: list[DirectiveNode] = field(default_factory=list)


@dataclass
class ParallelBlockNode(ASTNode):
    """Steps that execute concurrently.

    Example: PAR { @CFO FORECAST revenue; @CTO ASSESS infra }
    """

    branches: list[ASTNode] = field(default_factory=list)


@dataclass
class SequentialBlockNode(ASTNode):
    """Explicit sequential execution.

    Example: SEQ { @CDO GATHER data; @CFO ANALYZE $prev }
    """

    steps: list[ASTNode] = field(default_factory=list)


@dataclass
class ProgramNode(ASTNode):
    """Top-level container for a multi-statement program."""

    statements: list[ASTNode] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Condition nodes (kept for NL compiler and original parser compatibility)
# ---------------------------------------------------------------------------

@dataclass
class ExpressionNode(ASTNode):
    """A value reference: field:value or standalone value."""

    field: str = ""
    value: Optional[str] = None


@dataclass
class ConditionNode(ASTNode):
    """A condition: risk:high, cost > 1000, etc."""

    left: ExpressionNode = field(default_factory=ExpressionNode)
    comparator: str = ""  # ":", "==", ">", "<", ">=", "<=", "!="
    right: Optional[ExpressionNode] = None


@dataclass
class ConditionalNode(ASTNode):
    """IF condition THEN message [ELSE message]"""

    condition: ConditionNode | str = ""
    then_branch: ASTNode = field(default_factory=lambda: DirectiveNode())
    else_branch: Optional[ASTNode] = None
