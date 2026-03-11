"""Recursive descent parser for the agent communication grammar.

Parses token streams into AST nodes. Grammar:

    message     := conditional | pipeline
    pipeline    := directive ('|' directive)*
    directive   := [recipient] action target [params] [constraints] [output] [priority] [modifiers]
    conditional := 'IF' condition 'THEN' message ['ELSE' message]
"""

from __future__ import annotations

from prompt_optimizer.grammar.ast_nodes import (
    ASTNode,
    ConditionNode,
    ConditionalNode,
    ConstraintNode,
    ConstraintsNode,
    DirectiveNode,
    ExpressionNode,
    ModifierNode,
    OutputNode,
    ParamNode,
    ParamsNode,
    PipelineNode,
    PriorityNode,
    RecipientNode,
)
from prompt_optimizer.grammar.tokens import Token, TokenType


class ParseError(Exception):
    """Parse error with position information."""

    def __init__(self, message: str, token: Token | None = None):
        self.token = token
        pos = token.position if token else "?"
        super().__init__(f"Parse error at pos {pos}: {message}")


class Parser:
    """Recursive descent parser for agent communication grammar."""

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = [t for t in tokens if t.type != TokenType.NEWLINE]
        self._pos = 0

    def parse(self) -> ASTNode:
        """Parse the token stream into an AST."""
        node = self._message()
        # Don't require EOF — allow trailing identifiers
        return node

    # --- Helpers ---

    def _current(self) -> Token:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return Token(TokenType.EOF, "", -1)

    def _peek(self, offset: int = 0) -> Token:
        idx = self._pos + offset
        if idx < len(self._tokens):
            return self._tokens[idx]
        return Token(TokenType.EOF, "", -1)

    def _advance(self) -> Token:
        token = self._current()
        self._pos += 1
        return token

    def _expect(self, token_type: TokenType) -> Token:
        token = self._current()
        if token.type != token_type:
            raise ParseError(f"Expected {token_type.name}, got {token.type.name} ({token.value!r})", token)
        return self._advance()

    def _match(self, *types: TokenType) -> Token | None:
        if self._current().type in types:
            return self._advance()
        return None

    def _at_end(self) -> bool:
        return self._current().type == TokenType.EOF

    # --- Grammar rules ---

    def _message(self) -> ASTNode:
        """message := conditional | pipeline"""
        if self._current().type == TokenType.IF:
            return self._conditional()
        return self._pipeline()

    def _conditional(self) -> ConditionalNode:
        """conditional := 'IF' condition 'THEN' message ['ELSE' message]"""
        self._expect(TokenType.IF)
        condition = self._condition()
        self._expect(TokenType.THEN)
        then_branch = self._message()

        else_branch = None
        if self._match(TokenType.ELSE):
            else_branch = self._message()

        return ConditionalNode(
            condition=condition,
            then_branch=then_branch,
            else_branch=else_branch,
        )

    def _condition(self) -> ConditionNode:
        """condition := IDENTIFIER ':' IDENTIFIER | IDENTIFIER COMPARATOR value"""
        left_token = self._advance()
        left = ExpressionNode(field=left_token.value)

        # Shorthand: risk:high
        if self._match(TokenType.COLON):
            right_token = self._advance()
            left.value = None
            return ConditionNode(
                left=left,
                comparator=":",
                right=ExpressionNode(field=right_token.value),
            )

        # Comparator: cost > 1000
        comparator_types = {
            TokenType.GT: ">", TokenType.LT: "<",
            TokenType.GTE: ">=", TokenType.LTE: "<=",
            TokenType.EQ: "==", TokenType.NEQ: "!=",
        }

        for tt, op in comparator_types.items():
            if self._match(tt):
                right_token = self._advance()
                return ConditionNode(
                    left=left,
                    comparator=op,
                    right=ExpressionNode(field=right_token.value),
                )

        # No comparator — treat as boolean (field alone)
        return ConditionNode(left=left, comparator=":", right=None)

    def _pipeline(self) -> ASTNode:
        """pipeline := directive ('|' directive)*"""
        first = self._directive()
        directives = [first]

        while self._match(TokenType.PIPE):
            directives.append(self._directive())

        if len(directives) == 1:
            return directives[0]
        return PipelineNode(directives=directives)

    def _directive(self) -> DirectiveNode:
        """directive := [recipient] action target [params] [constraints] [output] [priority] [modifiers]"""
        recipient = self._recipient()
        action = self._action()
        target = self._target()
        params = self._params()
        constraints = self._constraints()
        output = self._output()
        priority = self._priority()
        modifiers = self._modifiers()

        return DirectiveNode(
            action=action,
            target=target,
            recipient=recipient,
            params=params,
            constraints=constraints,
            output=output,
            priority=priority,
            modifiers=modifiers,
        )

    def _recipient(self) -> RecipientNode | None:
        """recipient := '@' AGENT_CODE"""
        if self._current().type == TokenType.AT:
            self._advance()
            token = self._expect(TokenType.AGENT_CODE)
            return RecipientNode(agent_code=token.value)
        return None

    def _action(self) -> str:
        """action := ACTION"""
        token = self._current()
        if token.type == TokenType.ACTION:
            self._advance()
            return token.value
        # Allow identifiers as actions (for custom verbs)
        if token.type == TokenType.IDENTIFIER:
            self._advance()
            return token.value.upper()
        raise ParseError(f"Expected action verb, got {token.type.name} ({token.value!r})", token)

    def _target(self) -> str:
        """target := IDENTIFIER+ (collect words until a delimiter)"""
        parts: list[str] = []
        stop_types = {
            TokenType.LBRACE, TokenType.LBRACKET, TokenType.ARROW,
            TokenType.BANG, TokenType.TILDE, TokenType.PIPE, TokenType.AT,
            TokenType.EOF, TokenType.IF, TokenType.THEN, TokenType.ELSE,
        }

        while self._current().type not in stop_types:
            token = self._current()
            if token.type in (TokenType.IDENTIFIER, TokenType.NUMBER, TokenType.STRING, TokenType.AGENT_CODE, TokenType.ACTION):
                parts.append(token.value)
                self._advance()
            else:
                break

        return " ".join(parts) if parts else "unspecified"

    def _params(self) -> ParamsNode | None:
        """params := '{' param (',' param)* '}'"""
        if not self._match(TokenType.LBRACE):
            return None

        params: list[ParamNode] = []

        while self._current().type != TokenType.RBRACE and not self._at_end():
            # Skip commas
            self._match(TokenType.COMMA)

            token = self._current()
            if token.type == TokenType.RBRACE:
                break

            key = self._advance().value

            # Check for '=' value
            if self._match(TokenType.EQUALS):
                val_parts: list[str] = []
                while self._current().type not in (TokenType.COMMA, TokenType.RBRACE, TokenType.EOF):
                    val_parts.append(self._advance().value)
                params.append(ParamNode(key=key, value=" ".join(val_parts)))
            else:
                # Bare keyword like {inherit}
                params.append(ParamNode(key=key))

        self._expect(TokenType.RBRACE)
        return ParamsNode(params=params) if params else None

    def _constraints(self) -> ConstraintsNode | None:
        """constraints := '[' text (',' text)* ']'"""
        if not self._match(TokenType.LBRACKET):
            return None

        constraints: list[ConstraintNode] = []
        parts: list[str] = []

        while self._current().type != TokenType.RBRACKET and not self._at_end():
            if self._current().type == TokenType.COMMA:
                if parts:
                    constraints.append(ConstraintNode(text=" ".join(parts)))
                    parts = []
                self._advance()
            else:
                parts.append(self._advance().value)

        if parts:
            constraints.append(ConstraintNode(text=" ".join(parts)))

        self._expect(TokenType.RBRACKET)
        return ConstraintsNode(constraints=constraints) if constraints else None

    def _output(self) -> OutputNode | None:
        """output := '->' IDENTIFIER"""
        if not self._match(TokenType.ARROW):
            return None
        token = self._advance()
        return OutputNode(format=token.value)

    def _priority(self) -> PriorityNode | None:
        """priority := '!' IDENTIFIER"""
        if not self._match(TokenType.BANG):
            return None
        token = self._advance()
        return PriorityNode(level=token.value)

    def _modifiers(self) -> list[ModifierNode]:
        """modifiers := ('~' IDENTIFIER)*"""
        mods: list[ModifierNode] = []
        while self._match(TokenType.TILDE):
            token = self._advance()
            mods.append(ModifierNode(name=token.value))
        return mods
