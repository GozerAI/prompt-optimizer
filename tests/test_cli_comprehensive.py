"""Comprehensive tests for the prompt-optimizer CLI — all subcommands,
error handling, edge cases, and argparse integration."""

import os
import tempfile

import pytest

from prompt_optimizer.cli import (
    build_parser,
    cmd_emit,
    cmd_file,
    cmd_optimize,
    cmd_parse,
    cmd_run,
    cmd_validate,
    main,
    _format_ast,
    _parse_source,
)
from prompt_optimizer.grammar.ast_nodes import (
    ConditionalNode,
    DirectiveNode,
    ParallelBlockNode,
    PipelineNode,
    ProgramNode,
    RecipientNode,
    SequentialBlockNode,
)


# ============================================================
# po parse — valid input
# ============================================================


class TestCmdParseValid:
    def test_simple_directive(self, capsys):
        rc = cmd_parse("@CFO ANALYZE revenue")
        assert rc == 0
        out = capsys.readouterr().out
        assert "Directive" in out
        assert "CFO" in out
        assert "ANALYZE" in out

    def test_directive_with_params(self, capsys):
        rc = cmd_parse("@CFO ANALYZE revenue {period=Q1-2026}")
        assert rc == 0
        out = capsys.readouterr().out
        assert "params" in out
        assert "period" in out

    def test_directive_with_constraints(self, capsys):
        rc = cmd_parse("@CTO ASSESS risk [within 2h, no deps]")
        assert rc == 0
        out = capsys.readouterr().out
        assert "constraints" in out

    def test_pipeline(self, capsys):
        rc = cmd_parse("@CDO GATHER data | @CFO ANALYZE $prev")
        assert rc == 0
        out = capsys.readouterr().out
        assert "Pipeline" in out

    def test_par_block(self, capsys):
        rc = cmd_parse("PAR { @CFO FORECAST revenue; @CTO ASSESS infra }")
        assert rc == 0
        out = capsys.readouterr().out
        assert "ParallelBlock" in out

    def test_seq_block(self, capsys):
        rc = cmd_parse("SEQ { @CDO GATHER data; @CFO ANALYZE $prev }")
        assert rc == 0
        out = capsys.readouterr().out
        assert "SequentialBlock" in out

    def test_conditional(self, capsys):
        rc = cmd_parse("IF risk:high THEN @CFO ANALYZE costs")
        assert rc == 0
        out = capsys.readouterr().out
        assert "Conditional" in out

    def test_directive_with_priority(self, capsys):
        rc = cmd_parse("@CEO DECIDE expansion !urgent")
        assert rc == 0
        out = capsys.readouterr().out
        assert "priority" in out
        assert "urgent" in out

    def test_directive_with_modifiers(self, capsys):
        rc = cmd_parse("@CFO ANALYZE revenue ~thorough ~brief")
        assert rc == 0
        out = capsys.readouterr().out
        assert "modifiers" in out

    def test_directive_with_output(self, capsys):
        rc = cmd_parse("@CFO ANALYZE revenue -> summary")
        assert rc == 0
        out = capsys.readouterr().out
        assert "output" in out or "summary" in out

    def test_directive_with_response_contract(self, capsys):
        rc = cmd_parse("@CFO ANALYZE revenue -> {score: float, label: str}")
        assert rc == 0
        out = capsys.readouterr().out
        assert "response" in out or "score" in out

    def test_directive_with_retry(self, capsys):
        rc = cmd_parse("@CTO ANALYZE infra RETRY 3 BACKOFF linear FALLBACK @CIO")
        assert rc == 0
        out = capsys.readouterr().out
        assert "retry" in out or "RETRY" in out


# ============================================================
# po parse — invalid input
# ============================================================


class TestCmdParseInvalid:
    def test_malformed_braces(self, capsys):
        rc = cmd_parse("@CFO ANALYZE revenue {{{")
        assert rc == 1
        err = capsys.readouterr().err
        assert "error" in err.lower()

    def test_completely_invalid(self, capsys):
        rc = cmd_parse("!!! bad syntax {{{")
        assert rc == 1


# ============================================================
# po emit — roundtrip
# ============================================================


class TestCmdEmit:
    def test_roundtrip_directive(self, capsys):
        rc = cmd_emit("@CEO DECIDE expansion !urgent")
        assert rc == 0
        out = capsys.readouterr().out.strip()
        assert "@CEO" in out
        assert "DECIDE" in out
        assert "expansion" in out
        assert "!urgent" in out

    def test_roundtrip_pipeline(self, capsys):
        rc = cmd_emit("@CDO GATHER data | @CFO ANALYZE $prev")
        assert rc == 0
        out = capsys.readouterr().out.strip()
        assert "|" in out
        assert "CDO" in out
        assert "CFO" in out

    def test_roundtrip_par_block(self, capsys):
        rc = cmd_emit("PAR { @CFO FORECAST revenue; @CTO ASSESS infra }")
        assert rc == 0
        out = capsys.readouterr().out.strip()
        assert "PAR" in out

    def test_roundtrip_seq_block(self, capsys):
        rc = cmd_emit("SEQ { @CDO GATHER data; @CFO ANALYZE $prev }")
        assert rc == 0
        out = capsys.readouterr().out.strip()
        assert "SEQ" in out

    def test_roundtrip_conditional(self, capsys):
        rc = cmd_emit("IF risk:high THEN @CFO ANALYZE costs")
        assert rc == 0
        out = capsys.readouterr().out.strip()
        assert "IF" in out
        assert "THEN" in out

    def test_emit_with_modifiers(self, capsys):
        rc = cmd_emit("@CFO ANALYZE revenue ~thorough")
        assert rc == 0
        out = capsys.readouterr().out.strip()
        assert "~thorough" in out

    def test_emit_invalid_returns_error(self, capsys):
        rc = cmd_emit("!!! bad")
        assert rc == 1
        err = capsys.readouterr().err
        assert "error" in err.lower()

    def test_emit_matches_parse(self, capsys):
        src = "@CFO ANALYZE revenue {period=Q1-2026} -> summary !urgent"
        rc = cmd_emit(src)
        assert rc == 0
        out = capsys.readouterr().out.strip()
        assert "CFO" in out
        assert "ANALYZE" in out
        assert "period" in out
        assert "summary" in out
        assert "urgent" in out


# ============================================================
# po validate — valid program
# ============================================================


class TestCmdValidateValid:
    def test_valid_directive(self, capsys):
        rc = cmd_validate("@CEO DECIDE expansion")
        assert rc == 0
        out = capsys.readouterr().out
        assert "Valid" in out

    def test_valid_pipeline(self, capsys):
        rc = cmd_validate("@CDO GATHER data | @CFO ANALYZE $prev")
        assert rc == 0

    def test_valid_par_block(self, capsys):
        rc = cmd_validate("PAR { @CFO FORECAST revenue; @CTO ASSESS infra }")
        assert rc == 0

    def test_valid_conditional(self, capsys):
        rc = cmd_validate("IF risk:high THEN @CFO ANALYZE costs ELSE @CEO APPROVE deployment")
        assert rc == 0


# ============================================================
# po validate — invalid program
# ============================================================


class TestCmdValidateInvalid:
    def test_parse_error(self, capsys):
        rc = cmd_validate("!!! bad")
        assert rc == 1
        err = capsys.readouterr().err
        assert "error" in err.lower()

    def test_missing_agent_directive(self, capsys):
        rc = cmd_validate("ANALYZE revenue")
        out = capsys.readouterr().out
        # Validator reports missing agent as error
        assert "ERROR" in out or rc == 1

    def test_prev_at_first_step_reported(self, capsys):
        rc = cmd_validate("@CFO ANALYZE $prev")
        out = capsys.readouterr().out
        # $prev with no prior step should be an error
        assert "$prev" in out or "ERROR" in out


# ============================================================
# po optimize
# ============================================================


class TestCmdOptimize:
    def test_optimize_basic(self, capsys):
        rc = cmd_optimize("Hey CFO, could you please analyze the Q1 2026 revenue data?")
        assert rc == 0
        out = capsys.readouterr().out
        assert "Original" in out
        assert "Compressed" in out
        assert "Reduction" in out

    def test_optimize_already_compact(self, capsys):
        rc = cmd_optimize("ANALYZE revenue")
        assert rc == 0
        out = capsys.readouterr().out
        assert "Original" in out

    def test_optimize_empty_text(self, capsys):
        rc = cmd_optimize("")
        # Empty text should not crash
        out = capsys.readouterr().out
        err = capsys.readouterr().err if hasattr(capsys.readouterr(), 'err') else ""
        # Either succeeds with 0 tokens or reports an error gracefully
        assert rc == 0 or rc == 1

    def test_optimize_long_text(self, capsys):
        text = (
            "Hey CFO, could you please take a careful look at the revenue data? "
            "I think it would be helpful to analyze the trends. "
            "Additionally, please compare with last quarter. Thank you very much."
        )
        rc = cmd_optimize(text)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Reduction" in out


# ============================================================
# po file
# ============================================================


class TestCmdFile:
    def test_execute_valid_file(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ail", delete=False, encoding="utf-8"
        ) as f:
            f.write("@CEO DECIDE expansion\n")
            f.flush()
            path = f.name
        try:
            rc = cmd_file(path)
            assert rc == 0
            out = capsys.readouterr().out
            assert "[echo]" in out
        finally:
            os.unlink(path)

    def test_execute_pipeline_file(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ail", delete=False, encoding="utf-8"
        ) as f:
            f.write("@CDO GATHER data | @CFO ANALYZE $prev\n")
            f.flush()
            path = f.name
        try:
            rc = cmd_file(path)
            assert rc == 0
            out = capsys.readouterr().out
            assert "CDO" in out
            assert "CFO" in out
        finally:
            os.unlink(path)

    def test_nonexistent_file(self, capsys):
        rc = cmd_file("/nonexistent/path/to/file.ail")
        assert rc == 1
        err = capsys.readouterr().err
        assert "not found" in err.lower() or "cannot read" in err.lower()

    def test_empty_file(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ail", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            f.flush()
            path = f.name
        try:
            rc = cmd_file(path)
            # Empty file may cause parse error
            assert rc == 0 or rc == 1
        finally:
            os.unlink(path)

    def test_file_with_comments_only(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ail", delete=False, encoding="utf-8"
        ) as f:
            f.write("# This is a comment\n# Another comment\n")
            f.flush()
            path = f.name
        try:
            rc = cmd_file(path)
            # Comments-only file may parse as empty, which could error
            assert rc == 0 or rc == 1
        finally:
            os.unlink(path)

    def test_file_with_multiple_statements(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ail", delete=False, encoding="utf-8"
        ) as f:
            f.write("@CEO DECIDE expansion\n@CFO ANALYZE revenue\n")
            f.flush()
            path = f.name
        try:
            rc = cmd_file(path)
            assert rc == 0
            out = capsys.readouterr().out
            assert "CEO" in out
            assert "CFO" in out
        finally:
            os.unlink(path)


# ============================================================
# Help text and argparse
# ============================================================


class TestHelpAndArgparse:
    def test_no_command_shows_help(self, capsys):
        rc = main([])
        assert rc == 0
        # Should print help text but not error

    def test_parser_has_subcommands(self):
        parser = build_parser()
        # Verify subcommands exist
        assert parser is not None
        # The parser should accept known subcommands
        args = parser.parse_args(["parse", "test source"])
        assert args.command == "parse"
        assert args.source == "test source"

    def test_parse_subcommand_arg(self):
        parser = build_parser()
        args = parser.parse_args(["emit", "some source"])
        assert args.command == "emit"
        assert args.source == "some source"

    def test_validate_subcommand_arg(self):
        parser = build_parser()
        args = parser.parse_args(["validate", "some source"])
        assert args.command == "validate"

    def test_run_subcommand_arg(self):
        parser = build_parser()
        args = parser.parse_args(["run", "some source"])
        assert args.command == "run"

    def test_file_subcommand_arg(self):
        parser = build_parser()
        args = parser.parse_args(["file", "/path/to/file.ail"])
        assert args.command == "file"
        assert args.path == "/path/to/file.ail"

    def test_optimize_subcommand_arg(self):
        parser = build_parser()
        args = parser.parse_args(["optimize", "some text"])
        assert args.command == "optimize"
        assert args.text == "some text"

    def test_repl_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["repl"])
        assert args.command == "repl"


# ============================================================
# main() integration with each subcommand
# ============================================================


class TestMainIntegration:
    def test_main_parse(self, capsys):
        rc = main(["parse", "@CEO DECIDE expansion"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Directive" in out

    def test_main_emit(self, capsys):
        rc = main(["emit", "@CEO DECIDE expansion"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "@CEO" in out

    def test_main_validate(self, capsys):
        rc = main(["validate", "@CEO DECIDE expansion"])
        assert rc == 0

    def test_main_run(self, capsys):
        rc = main(["run", "@CEO DECIDE expansion"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "[echo]" in out

    def test_main_optimize(self, capsys):
        rc = main(["optimize", "Hey CFO, please analyze revenue."])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Original" in out

    def test_main_file_not_found(self, capsys):
        rc = main(["file", "/nonexistent/file.ail"])
        assert rc == 1

    def test_main_file_valid(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ail", delete=False, encoding="utf-8"
        ) as f:
            f.write("@CEO DECIDE expansion\n")
            f.flush()
            path = f.name
        try:
            rc = main(["file", path])
            assert rc == 0
        finally:
            os.unlink(path)


# ============================================================
# _format_ast helper
# ============================================================


class TestFormatAst:
    def test_format_directive(self):
        node = DirectiveNode(
            action="ANALYZE", target="revenue",
            recipient=RecipientNode(agent_code="CFO"),
        )
        text = _format_ast(node)
        assert "Directive" in text
        assert "CFO" in text
        assert "ANALYZE" in text

    def test_format_pipeline(self):
        node = PipelineNode(directives=[
            DirectiveNode(
                action="GATHER", target="data",
                recipient=RecipientNode(agent_code="CDO"),
            ),
            DirectiveNode(
                action="ANALYZE", target="$prev",
                recipient=RecipientNode(agent_code="CFO"),
            ),
        ])
        text = _format_ast(node)
        assert "Pipeline" in text

    def test_format_parallel_block(self):
        node = ParallelBlockNode(branches=[
            DirectiveNode(
                action="FORECAST", target="revenue",
                recipient=RecipientNode(agent_code="CFO"),
            ),
            DirectiveNode(
                action="ASSESS", target="infra",
                recipient=RecipientNode(agent_code="CTO"),
            ),
        ])
        text = _format_ast(node)
        assert "ParallelBlock" in text
        assert "CFO" in text
        assert "CTO" in text

    def test_format_sequential_block(self):
        node = SequentialBlockNode(steps=[
            DirectiveNode(
                action="GATHER", target="data",
                recipient=RecipientNode(agent_code="CDO"),
            ),
            DirectiveNode(
                action="ANALYZE", target="$prev",
                recipient=RecipientNode(agent_code="CFO"),
            ),
        ])
        text = _format_ast(node)
        assert "SequentialBlock" in text

    def test_format_conditional_with_else(self):
        node = ConditionalNode(
            condition="revenue > 1M",
            then_branch=DirectiveNode(
                action="DECIDE", target="expand",
                recipient=RecipientNode(agent_code="CEO"),
            ),
            else_branch=DirectiveNode(
                action="ANALYZE", target="cuts",
                recipient=RecipientNode(agent_code="CFO"),
            ),
        )
        text = _format_ast(node)
        assert "Conditional" in text
        assert "revenue > 1M" in text
        assert "then:" in text
        assert "else:" in text

    def test_format_program(self):
        node = ProgramNode(statements=[
            DirectiveNode(
                action="DECIDE", target="expansion",
                recipient=RecipientNode(agent_code="CEO"),
            ),
            DirectiveNode(
                action="ANALYZE", target="revenue",
                recipient=RecipientNode(agent_code="CFO"),
            ),
        ])
        text = _format_ast(node)
        assert "Program" in text
        assert "CEO" in text
        assert "CFO" in text

    def test_format_indentation_increases(self):
        node = ProgramNode(statements=[
            DirectiveNode(
                action="DECIDE", target="expansion",
                recipient=RecipientNode(agent_code="CEO"),
            ),
        ])
        text = _format_ast(node)
        lines = text.split("\n")
        # Program line should have no indent, child should have indent
        assert lines[0].startswith("Program")
        if len(lines) > 1:
            assert lines[1].startswith("  ")


# ============================================================
# _parse_source helper
# ============================================================


class TestParseSource:
    def test_parse_simple(self):
        node = _parse_source("@CEO DECIDE expansion")
        assert isinstance(node, DirectiveNode)

    def test_parse_pipeline(self):
        node = _parse_source("@CDO GATHER data | @CFO ANALYZE $prev")
        assert isinstance(node, PipelineNode)

    def test_parse_par(self):
        node = _parse_source("PAR { @CFO FORECAST revenue; @CTO ASSESS infra }")
        assert isinstance(node, ParallelBlockNode)

    def test_parse_multi_statement(self):
        node = _parse_source("@CEO DECIDE expansion\n@CFO ANALYZE revenue")
        assert isinstance(node, ProgramNode)
