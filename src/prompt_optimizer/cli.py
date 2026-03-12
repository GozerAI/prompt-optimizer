"""CLI and REPL for the prompt-optimizer grammar."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from prompt_optimizer.grammar.ast_nodes import (
    ASTNode,
    ConditionalNode,
    DirectiveNode,
    ParallelBlockNode,
    PipelineNode,
    ProgramNode,
    SequentialBlockNode,
)
from prompt_optimizer.grammar.lexer import Lexer
from prompt_optimizer.grammar.parser import ParseError, Parser
from prompt_optimizer.grammar.renderer import Renderer
from prompt_optimizer.grammar.validator import Validator
from prompt_optimizer.runtime.context import ExecutionContext
from prompt_optimizer.runtime.executor import AgentAdapter, Executor


class EchoAdapter(AgentAdapter):
    """Mock adapter that prints what each agent would do and returns a dict echo."""

    async def execute_directive(
        self,
        agent: str,
        action: str,
        target: Any,
        params: dict[str, Any],
        constraints: list[str],
        context: ExecutionContext,
    ) -> Any:
        parts = [f"@{agent} {action}"]
        if target:
            parts.append(f"target={target!r}")
        if params:
            parts.append(f"params={params!r}")
        if constraints:
            parts.append(f"constraints={constraints!r}")
        print(f"  [echo] {' '.join(parts)}")
        return {"agent": agent, "action": action, "status": "echo"}

    async def evaluate_condition(self, condition: str, context: ExecutionContext) -> bool:
        print(f"  [echo] IF {condition} => True (default)")
        return True


def _parse_source(source: str) -> ASTNode:
    """Parse source string into AST."""
    tokens = Lexer().tokenize(source)
    return Parser(tokens).parse()


def _format_ast(node: ASTNode, indent: int = 0) -> str:
    """Pretty-print an AST node."""
    prefix = "  " * indent
    if isinstance(node, ProgramNode):
        lines = [f"{prefix}Program"]
        for stmt in node.statements:
            lines.append(_format_ast(stmt, indent + 1))
        return "\n".join(lines)
    if isinstance(node, DirectiveNode):
        agent = node.recipient.agent_code if node.recipient else "?"
        parts = [f"{prefix}Directive @{agent} {node.action}"]
        if node.target:
            parts.append(f"{prefix}  target: {node.target!r}")
        if node.params and node.params.params:
            pdict = {p.key: p.value for p in node.params.params}
            parts.append(f"{prefix}  params: {pdict!r}")
        if node.constraints and node.constraints.constraints:
            clist = [c.text for c in node.constraints.constraints]
            parts.append(f"{prefix}  constraints: {clist!r}")
        if node.response_contract:
            parts.append(f"{prefix}  response: {node.response_contract!r}")
        elif node.output:
            parts.append(f"{prefix}  output: {node.output.format}")
        if node.priority:
            parts.append(f"{prefix}  priority: {node.priority.level}")
        if node.modifiers:
            parts.append(f"{prefix}  modifiers: {[m.name for m in node.modifiers]!r}")
        if node.retry:
            parts.append(f"{prefix}  retry: {node.retry!r}")
        return "\n".join(parts)
    if isinstance(node, PipelineNode):
        lines = [f"{prefix}Pipeline"]
        for step in node.directives:
            lines.append(_format_ast(step, indent + 1))
        return "\n".join(lines)
    if isinstance(node, ParallelBlockNode):
        lines = [f"{prefix}ParallelBlock"]
        for branch in node.branches:
            lines.append(_format_ast(branch, indent + 1))
        return "\n".join(lines)
    if isinstance(node, SequentialBlockNode):
        lines = [f"{prefix}SequentialBlock"]
        for step in node.steps:
            lines.append(_format_ast(step, indent + 1))
        return "\n".join(lines)
    if isinstance(node, ConditionalNode):
        lines = [f"{prefix}Conditional: {node.condition}"]
        lines.append(f"{prefix}  then:")
        lines.append(_format_ast(node.then_branch, indent + 2))
        if node.else_branch:
            lines.append(f"{prefix}  else:")
            lines.append(_format_ast(node.else_branch, indent + 2))
        return "\n".join(lines)
    return f"{prefix}{node!r}"


def cmd_parse(source: str) -> int:
    """Parse and pretty-print the AST."""
    try:
        node = _parse_source(source)
        print(_format_ast(node))
        return 0
    except (SyntaxError, ParseError) as e:
        print(f"Parse error: {e}", file=sys.stderr)
        return 1


def cmd_emit(source: str) -> int:
    """Parse then emit back to wire format (roundtrip check)."""
    try:
        node = _parse_source(source)
        print(Renderer().render(node))
        return 0
    except (SyntaxError, ParseError) as e:
        print(f"Parse error: {e}", file=sys.stderr)
        return 1


def cmd_validate(source: str) -> int:
    """Run validator and print errors/warnings."""
    try:
        node = _parse_source(source)
    except (SyntaxError, ParseError) as e:
        print(f"Parse error: {e}", file=sys.stderr)
        return 1

    result = Validator().validate(node)
    if result.errors:
        for err in result.errors:
            print(f"ERROR: {err}")
    if result.warnings:
        for warn in result.warnings:
            print(f"WARNING: {warn}")
    if result.valid and not result.warnings:
        print("Valid.")
    return 0 if result.valid else 1


def cmd_run(source: str) -> int:
    """Execute with echo adapter."""
    try:
        node = _parse_source(source)
    except (SyntaxError, ParseError) as e:
        print(f"Parse error: {e}", file=sys.stderr)
        return 1

    adapter = EchoAdapter()
    executor = Executor(adapter)

    try:
        result = asyncio.run(executor.execute(node))
        print(f"Result: {result!r}")
        return 0
    except Exception as e:
        print(f"Execution error: {e}", file=sys.stderr)
        return 1


def cmd_file(path: str) -> int:
    """Load and execute a workflow file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
    except FileNotFoundError:
        print(f"File not found: {path}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"Cannot read file: {e}", file=sys.stderr)
        return 1

    return cmd_run(source)


def cmd_optimize(text: str) -> int:
    """Run 3-layer prompt compression."""
    try:
        from prompt_optimizer import optimize
        result = optimize(text)
        print(f"Original:   {result.token_counts.original} tokens")
        print(f"Compressed: {result.token_counts.compressed} tokens")
        print(f"Reduction:  {result.token_counts.reduction_pct * 100:.1f}%")
        print(f"Layers:     {result.layers_applied}")
        print()
        print(result.compressed_text)
        return 0
    except Exception as e:
        print(f"Optimization error: {e}", file=sys.stderr)
        return 1


def cmd_repl() -> int:
    """Interactive REPL."""
    print("Prompt Optimizer REPL (type 'quit' or 'exit' to leave, prefix with '!' to execute)")
    adapter = EchoAdapter()
    executor = Executor(adapter)

    while True:
        try:
            line = input("po> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        line = line.strip()
        if not line:
            continue
        if line.lower() in ("quit", "exit"):
            break

        execute = False
        if line.startswith("!"):
            execute = True
            line = line[1:].strip()

        try:
            node = _parse_source(line)
        except (SyntaxError, ParseError) as e:
            print(f"Parse error: {e}")
            continue

        if execute:
            try:
                result = asyncio.run(executor.execute(node))
                print(f"Result: {result!r}")
            except Exception as e:
                print(f"Execution error: {e}")
        else:
            result = Validator().validate(node)
            if result.errors:
                for err in result.errors:
                    print(f"ERROR: {err}")
            if result.warnings:
                for warn in result.warnings:
                    print(f"WARNING: {warn}")
            if result.valid:
                print(_format_ast(node))

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser."""
    parser = argparse.ArgumentParser(
        prog="po",
        description="Prompt Optimizer CLI — parse, validate, execute, and compress agent programs",
    )
    sub = parser.add_subparsers(dest="command")

    p_parse = sub.add_parser("parse", help="Parse source and pretty-print the AST")
    p_parse.add_argument("source", help="Source string")

    p_emit = sub.add_parser("emit", help="Parse then emit back to wire format")
    p_emit.add_argument("source", help="Source string")

    p_validate = sub.add_parser("validate", help="Run validator and print errors/warnings")
    p_validate.add_argument("source", help="Source string")

    p_run = sub.add_parser("run", help="Execute with echo adapter")
    p_run.add_argument("source", help="Source string")

    sub.add_parser("repl", help="Interactive REPL")

    p_file = sub.add_parser("file", help="Load and execute a workflow file")
    p_file.add_argument("path", help="Path to workflow file")

    p_optimize = sub.add_parser("optimize", help="Run 3-layer prompt compression")
    p_optimize.add_argument("text", help="Text to optimize")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "parse": lambda: cmd_parse(args.source),
        "emit": lambda: cmd_emit(args.source),
        "validate": lambda: cmd_validate(args.source),
        "run": lambda: cmd_run(args.source),
        "repl": cmd_repl,
        "file": lambda: cmd_file(args.path),
        "optimize": lambda: cmd_optimize(args.text),
    }

    handler = commands.get(args.command)
    if handler:
        return handler()

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
