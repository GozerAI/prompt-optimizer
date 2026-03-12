"""Recursive descent parser for the agent communication grammar.

Parses token streams into AST nodes. Supports both simple directive notation
and full orchestration syntax (PAR/SEQ blocks, pipelines, contracts, retry).

Grammar:
    program     := statement ((NEWLINE | SEMICOLON) statement)* EOF
    statement   := conditional | par_block | seq_block | pipeline
    par_block   := PAR '{' statement (';' statement)* '}'
    seq_block   := SEQ '{' statement (';' statement)* '}'
    pipeline    := directive ('|' directive)*
    directive   := [recipient] action target [params] [constraints] [bb_refs]
                   [output|contract] [priority] [modifiers] [retry]
    conditional := 'IF' condition 'THEN' statement ['ELSE' statement]
"""

from __future__ import annotations

import re

from prompt_optimizer.grammar.ast_nodes import (
    ASTNode,
    AgentFieldRefNode,
    BlackboardRefNode,
    ConditionNode,
    ConditionalNode,
    ConstraintNode,
    ConstraintsNode,
    ContractFieldNode,
    DirectiveNode,
    ExpressionNode,
    ModifierNode,
    OutputNode,
    ParallelBlockNode,
    ParamNode,
    ParamsNode,
    PipelineNode,
    PrevRefNode,
    PriorityNode,
    ProgramNode,
    RecipientNode,
    ResponseContractNode,
    RetryPolicyNode,
    SequentialBlockNode,
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
        # Keep newlines for program-level multi-statement parsing
        self._tokens = tokens
        self._pos = 0

    def parse(self) -> ASTNode:
        """Parse the token stream into an AST."""
        self._skip_newlines()
        stmts: list[ASTNode] = []
        stmts.append(self._statement())

        while True:
            self._skip_newlines()
            if self._current().type == TokenType.EOF:
                break
            if self._match(TokenType.SEMICOLON):
                self._skip_newlines()
                if self._current().type == TokenType.EOF:
                    break
                stmts.append(self._statement())
            elif self._current().type in (
                TokenType.AT, TokenType.AGENT_CODE, TokenType.AGENT_FIELD,
                TokenType.PAR, TokenType.SEQ, TokenType.IF,
            ):
                # Newline-separated statements
                stmts.append(self._statement())
            else:
                break

        if len(stmts) == 1:
            return stmts[0]
        return ProgramNode(statements=stmts)

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

    def _skip_newlines(self) -> None:
        while self._current().type == TokenType.NEWLINE:
            self._advance()

    # --- Grammar rules ---

    def _statement(self) -> ASTNode:
        """statement := conditional | par_block | seq_block | pipeline"""
        self._skip_newlines()
        tok = self._current()
        if tok.type == TokenType.IF:
            return self._conditional()
        if tok.type == TokenType.PAR:
            return self._par_block()
        if tok.type == TokenType.SEQ:
            return self._seq_block()
        return self._pipeline()

    def _conditional(self) -> ConditionalNode:
        """conditional := 'IF' condition 'THEN' statement ['ELSE' statement]"""
        self._expect(TokenType.IF)

        # Try structured condition first (for backward compatibility with NL compiler)
        condition = self._try_structured_condition()
        if condition is None:
            # Fall back to raw condition string (AIL style)
            condition = self._raw_condition()

        self._expect(TokenType.THEN)
        then_branch = self._statement()

        else_branch = None
        if self._match(TokenType.ELSE):
            else_branch = self._statement()

        return ConditionalNode(
            condition=condition,
            then_branch=then_branch,
            else_branch=else_branch,
        )

    def _try_structured_condition(self) -> ConditionNode | None:
        """Try to parse a structured condition (field:value or field CMP value)."""
        saved = self._pos
        try:
            left_token = self._current()
            if left_token.type not in (TokenType.IDENTIFIER, TokenType.AGENT_CODE, TokenType.ACTION):
                return None
            self._advance()
            left = ExpressionNode(field=left_token.value)

            # Shorthand: risk:high
            if self._match(TokenType.COLON):
                right_token = self._advance()
                return ConditionNode(
                    left=left,
                    comparator=":",
                    right=ExpressionNode(field=right_token.value),
                )

            # Comparator
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

            # No comparator — boolean
            return ConditionNode(left=left, comparator=":", right=None)
        except Exception:
            self._pos = saved
            return None

    def _raw_condition(self) -> str:
        """Consume tokens until THEN, collecting as raw condition string."""
        parts: list[str] = []
        while self._current().type not in (TokenType.THEN, TokenType.EOF):
            tok = self._advance()
            parts.append(tok.value)
        return " ".join(parts)

    def _par_block(self) -> ParallelBlockNode:
        """par_block := PAR '{' statement (';' statement)* '}'"""
        self._expect(TokenType.PAR)
        self._expect(TokenType.LBRACE)
        self._skip_newlines()
        branches: list[ASTNode] = [self._statement()]
        while True:
            self._skip_newlines()
            if self._current().type == TokenType.RBRACE:
                break
            if self._match(TokenType.SEMICOLON):
                self._skip_newlines()
                if self._current().type == TokenType.RBRACE:
                    break
                branches.append(self._statement())
            else:
                break
        self._expect(TokenType.RBRACE)
        return ParallelBlockNode(branches=branches)

    def _seq_block(self) -> SequentialBlockNode:
        """seq_block := SEQ '{' statement (';' statement)* '}'"""
        self._expect(TokenType.SEQ)
        self._expect(TokenType.LBRACE)
        self._skip_newlines()
        steps: list[ASTNode] = [self._statement()]
        while True:
            self._skip_newlines()
            if self._current().type == TokenType.RBRACE:
                break
            if self._match(TokenType.SEMICOLON):
                self._skip_newlines()
                if self._current().type == TokenType.RBRACE:
                    break
                steps.append(self._statement())
            else:
                break
        self._expect(TokenType.RBRACE)
        return SequentialBlockNode(steps=steps)

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
        """directive := [recipient] action target [params] [constraints] [bb_refs]
                        [output|contract] [priority] [modifiers] [retry]"""
        recipient = self._recipient()
        action = self._action()
        target = self._target()
        params = self._params()

        # Constraints and/or bb refs in brackets
        constraints = None
        context_refs: list[BlackboardRefNode] = []
        if self._current().type == TokenType.LBRACKET:
            constraints, context_refs = self._bracket_content()

        # Second bracket group (could be bb refs if first was constraints)
        if self._current().type == TokenType.LBRACKET and not context_refs:
            _, context_refs = self._bracket_content()

        # Response: output format or contract
        output = None
        response_contract = None
        if self._current().type == TokenType.ARROW:
            output, response_contract = self._response()

        # Priority (! syntax or PRIORITY token)
        priority = self._priority()

        # Modifiers
        modifiers = self._modifiers()

        # Retry
        retry = None
        if self._current().type == TokenType.RETRY:
            retry = self._retry()

        return DirectiveNode(
            action=action,
            target=target,
            recipient=recipient,
            params=params,
            constraints=constraints,
            output=output,
            priority=priority,
            modifiers=modifiers,
            retry=retry,
            response_contract=response_contract,
            context_refs=context_refs,
        )

    def _recipient(self) -> RecipientNode | None:
        """recipient := '@' AGENT_CODE"""
        if self._current().type == TokenType.AT:
            self._advance()
            token = self._expect(TokenType.AGENT_CODE)
            return RecipientNode(agent_code=token.value)
        return None

    def _action(self) -> str:
        """action := ACTION | IDENTIFIER"""
        token = self._current()
        if token.type == TokenType.ACTION:
            self._advance()
            return token.value
        if token.type == TokenType.IDENTIFIER:
            self._advance()
            return token.value.upper()
        raise ParseError(f"Expected action verb, got {token.type.name} ({token.value!r})", token)

    def _target(self) -> str | PrevRefNode | BlackboardRefNode | AgentFieldRefNode:
        """target := PREV_REF | BB_REF | AGENT_FIELD | IDENTIFIER+ | STRING | NUMBER"""
        tok = self._current()

        if tok.type == TokenType.PREV_REF:
            return self._prev_ref()
        if tok.type == TokenType.BB_REF:
            return self._bb_ref()
        if tok.type == TokenType.AGENT_FIELD:
            return self._agent_field_ref()

        # Collect word tokens as target
        stop_types = {
            TokenType.LBRACE, TokenType.LBRACKET, TokenType.ARROW,
            TokenType.BANG, TokenType.TILDE, TokenType.PIPE, TokenType.AT,
            TokenType.EOF, TokenType.IF, TokenType.THEN, TokenType.ELSE,
            TokenType.SEMICOLON, TokenType.RBRACE, TokenType.NEWLINE,
            TokenType.RETRY, TokenType.PAR, TokenType.SEQ,
            TokenType.PRIORITY, TokenType.MODIFIER,
        }

        parts: list[str] = []
        while self._current().type not in stop_types:
            token = self._current()
            if token.type in (TokenType.IDENTIFIER, TokenType.NUMBER, TokenType.STRING,
                              TokenType.AGENT_CODE, TokenType.ACTION):
                parts.append(token.value)
                self._advance()
            else:
                break

        return " ".join(parts) if parts else ""

    def _params(self) -> ParamsNode | None:
        """params := '{' param (',' param)* '}'"""
        if not self._match(TokenType.LBRACE):
            return None

        params: list[ParamNode] = []

        while self._current().type != TokenType.RBRACE and not self._at_end():
            self._match(TokenType.COMMA)

            token = self._current()
            if token.type == TokenType.RBRACE:
                break

            key = self._advance().value

            if self._match(TokenType.EQUALS):
                val_parts: list[str] = []
                while self._current().type not in (TokenType.COMMA, TokenType.RBRACE, TokenType.EOF):
                    val_parts.append(self._advance().value)
                params.append(ParamNode(key=key, value=" ".join(val_parts)))
            else:
                params.append(ParamNode(key=key))

        self._expect(TokenType.RBRACE)
        return ParamsNode(params=params) if params else None

    def _bracket_content(self) -> tuple[ConstraintsNode | None, list[BlackboardRefNode]]:
        """Parse [...] which can contain constraints or bb refs."""
        self._expect(TokenType.LBRACKET)
        constraints: list[ConstraintNode] = []
        refs: list[BlackboardRefNode] = []

        while self._current().type != TokenType.RBRACKET and not self._at_end():
            if self._current().type == TokenType.BB_REF:
                refs.append(self._bb_ref())
            else:
                parts: list[str] = []
                while self._current().type not in (
                    TokenType.COMMA, TokenType.RBRACKET, TokenType.EOF
                ):
                    parts.append(self._advance().value)
                if parts:
                    constraints.append(ConstraintNode(text=" ".join(parts)))
            self._match(TokenType.COMMA)

        self._expect(TokenType.RBRACKET)
        c_node = ConstraintsNode(constraints=constraints) if constraints else None
        return c_node, refs

    def _response(self) -> tuple[OutputNode | None, ResponseContractNode | None]:
        """Parse -> format_hint or -> {field: type, ...}"""
        self._expect(TokenType.ARROW)

        # Contract: -> {field: type, ...}
        if self._current().type == TokenType.LBRACE:
            contract = self._contract()
            return None, contract

        # Format hint: -> summary
        if self._current().type in (TokenType.IDENTIFIER, TokenType.ACTION):
            hint = self._advance().value.lower()
            return OutputNode(format=hint), None

        # Allow it to pass through
        return None, None

    def _contract(self) -> ResponseContractNode:
        """Parse response contract: {field: type, field?: type, ...}"""
        self._expect(TokenType.LBRACE)
        fields: list[ContractFieldNode] = []
        if self._current().type != TokenType.RBRACE:
            fields.append(self._contract_field())
            while self._match(TokenType.COMMA):
                fields.append(self._contract_field())
        self._expect(TokenType.RBRACE)
        return ResponseContractNode(fields=tuple(fields))

    def _contract_field(self) -> ContractFieldNode:
        """Parse a contract field: name[?]: type"""
        name = self._expect(TokenType.IDENTIFIER).value
        required = True
        if self._match(TokenType.QUESTION):
            required = False
        self._expect(TokenType.COLON)
        type_hint = self._expect(TokenType.IDENTIFIER).value
        return ContractFieldNode(name=name, type_hint=type_hint, required=required)

    def _priority(self) -> PriorityNode | None:
        """priority := '!' IDENTIFIER | PRIORITY token"""
        if self._current().type == TokenType.PRIORITY:
            tok = self._advance()
            return PriorityNode(level=tok.value)
        if self._match(TokenType.BANG):
            token = self._advance()
            return PriorityNode(level=token.value)
        return None

    def _modifiers(self) -> list[ModifierNode]:
        """modifiers := ('~' IDENTIFIER)* | MODIFIER*"""
        mods: list[ModifierNode] = []
        while True:
            if self._current().type == TokenType.MODIFIER:
                mods.append(ModifierNode(name=self._advance().value))
            elif self._match(TokenType.TILDE):
                token = self._advance()
                mods.append(ModifierNode(name=token.value))
            else:
                break
        return mods

    def _retry(self) -> RetryPolicyNode:
        """retry := RETRY NUMBER [BACKOFF IDENT] [FALLBACK '@' AGENT]"""
        self._expect(TokenType.RETRY)
        max_retries = int(self._expect(TokenType.NUMBER).value)
        backoff = "exp"
        fallback_agent = None
        if self._match(TokenType.BACKOFF):
            backoff = self._expect(TokenType.IDENTIFIER).value
        if self._match(TokenType.FALLBACK):
            self._expect(TokenType.AT)
            fallback_agent = self._expect(TokenType.AGENT_CODE).value
        return RetryPolicyNode(
            max_retries=max_retries,
            backoff=backoff,
            fallback_agent=fallback_agent,
        )

    # --- Ref parsing ---

    def _prev_ref(self) -> PrevRefNode:
        tok = self._expect(TokenType.PREV_REF)
        m = re.search(r"\[(\d+)\]", tok.value)
        step = int(m.group(1)) if m else None
        return PrevRefNode(step=step)

    def _bb_ref(self) -> BlackboardRefNode:
        tok = self._expect(TokenType.BB_REF)
        m = re.match(r"bb:([a-z_][a-z0-9_]*):([a-z_][a-z0-9_]*)(?:@v(\d+))?", tok.value)
        if not m:
            raise ParseError(f"Invalid blackboard ref: {tok.value}", tok)
        ns, key = m.group(1), m.group(2)
        ver = int(m.group(3)) if m.group(3) else None
        return BlackboardRefNode(namespace=ns, key=key, version=ver)

    def _agent_field_ref(self) -> AgentFieldRefNode:
        tok = self._expect(TokenType.AGENT_FIELD)
        m = re.match(r"@([A-Z][A-Za-z0-9_]*)\.([a-z_][a-z0-9_]*)", tok.value)
        if not m:
            raise ParseError(f"Invalid agent field ref: {tok.value}", tok)
        return AgentFieldRefNode(agent=m.group(1), field=m.group(2))
