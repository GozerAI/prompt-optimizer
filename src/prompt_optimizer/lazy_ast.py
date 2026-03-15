"""Lazy AST node evaluation.

Wraps AST nodes with deferred evaluation -- child nodes are only
fully processed when accessed."""

from __future__ import annotations

from typing import Any, Callable

from prompt_optimizer.grammar.ast_nodes import (
    ASTNode,
    ConditionalNode,
    DirectiveNode,
    ParallelBlockNode,
    PipelineNode,
    ProgramNode,
    SequentialBlockNode,
)


class LazyNode:
    """Wraps an AST node with lazy evaluation.

    The wrapped node is computed from a factory function on first access.
    Subsequent accesses return the cached result."""

    __slots__ = ("_factory", "_cached", "_evaluated", "node_type")

    def __init__(self, factory: Callable[[], ASTNode], node_type: str = "") -> None:
        self._factory = factory
        self._cached: ASTNode | None = None
        self._evaluated = False
        self.node_type = node_type

    @property
    def value(self) -> ASTNode:
        """Evaluate and return the wrapped node."""
        if not self._evaluated:
            self._cached = self._factory()
            self._evaluated = True
        return self._cached  # type: ignore[return-value]

    @property
    def is_evaluated(self) -> bool:
        return self._evaluated

    def invalidate(self) -> None:
        """Force re-evaluation on next access."""
        self._evaluated = False
        self._cached = None


class LazyProgram:
    """A program node with lazily-evaluated statements.

    Each statement is only parsed/evaluated when accessed by index."""

    def __init__(self, statement_factories: list[Callable[[], ASTNode]]) -> None:
        self._factories = statement_factories
        self._cache: dict[int, ASTNode] = {}

    def __len__(self) -> int:
        return len(self._factories)

    def __getitem__(self, index: int) -> ASTNode:
        if index not in self._cache:
            if index < 0 or index >= len(self._factories):
                raise IndexError(f"Statement index {index} out of range")
            self._cache[index] = self._factories[index]()
        return self._cache[index]

    def get_evaluated_count(self) -> int:
        """How many statements have been evaluated so far."""
        return len(self._cache)

    def to_program_node(self) -> ProgramNode:
        """Force-evaluate all statements and return a ProgramNode."""
        stmts = [self[i] for i in range(len(self._factories))]
        return ProgramNode(statements=stmts)


class LazyParser:
    """Wraps the standard parser to produce lazy AST nodes.

    Statements in a ProgramNode are parsed lazily -- only tokenized
    and parsed when actually accessed."""

    def __init__(self) -> None:
        from prompt_optimizer.grammar.lexer import Lexer
        self._lexer = Lexer()

    def parse_lazy(self, text: str) -> LazyProgram:
        """Parse text into a LazyProgram with deferred statement evaluation."""
        from prompt_optimizer.grammar.parser import Parser

        lines = [line.strip() for line in text.split("
") if line.strip()]
        if not lines:
            return LazyProgram([])

        def make_factory(line_text):
            def factory():
                tokens = self._lexer.tokenize(line_text)
                return Parser(tokens).parse()
            return factory

        factories = [make_factory(line) for line in lines]
        return LazyProgram(factories)
