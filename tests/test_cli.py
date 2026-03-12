"""Tests for the prompt-optimizer CLI (ported from AIL)."""

import os
import tempfile

import pytest

from prompt_optimizer.cli import main, cmd_parse, cmd_emit, cmd_validate, cmd_run, cmd_file, _format_ast, _parse_source
from prompt_optimizer.grammar.ast_nodes import (
    ConditionalNode,
    DirectiveNode,
    ParallelBlockNode,
    RecipientNode,
)


# ---------------------------------------------------------------------------
# Unit tests: direct function calls
# ---------------------------------------------------------------------------

class TestCmdParse:
    def test_simple_directive(self, capsys):
        rc = cmd_parse("@CEO DECIDE expansion")
        assert rc == 0
        out = capsys.readouterr().out
        assert "Directive" in out
        assert "CEO" in out
        assert "DECIDE" in out

    def test_pipeline(self, capsys):
        rc = cmd_parse("@CDO GATHER data | @CFO ANALYZE $prev")
        assert rc == 0
        out = capsys.readouterr().out
        assert "Pipeline" in out

    def test_invalid_source(self, capsys):
        rc = cmd_parse("not valid ail at all {{{")
        # Should error on parse
        assert rc == 1


class TestCmdEmit:
    def test_roundtrip_directive(self, capsys):
        rc = cmd_emit("@CEO DECIDE expansion !urgent")
        assert rc == 0
        out = capsys.readouterr().out.strip()
        assert "@CEO DECIDE expansion" in out
        assert "!urgent" in out

    def test_roundtrip_pipeline(self, capsys):
        rc = cmd_emit("@CDO GATHER data | @CFO ANALYZE $prev")
        assert rc == 0
        out = capsys.readouterr().out.strip()
        assert "|" in out

    def test_invalid_source(self, capsys):
        rc = cmd_emit("!!! bad")
        assert rc == 1


class TestCmdValidate:
    def test_valid_directive(self, capsys):
        rc = cmd_validate("@CEO DECIDE expansion")
        assert rc == 0
        out = capsys.readouterr().out
        assert "Valid" in out

    def test_default_no_warnings(self, capsys):
        rc = cmd_validate("@CEO DECIDE expansion")
        assert rc == 0

    def test_invalid_source(self, capsys):
        rc = cmd_validate("!!! bad")
        assert rc == 1


class TestCmdRun:
    def test_echo_simple(self, capsys):
        rc = cmd_run("@CEO DECIDE expansion")
        assert rc == 0
        out = capsys.readouterr().out
        assert "[echo]" in out
        assert "CEO" in out
        assert "Result:" in out

    def test_echo_pipeline(self, capsys):
        rc = cmd_run("@CDO GATHER data | @CFO ANALYZE $prev")
        assert rc == 0
        out = capsys.readouterr().out
        assert "CDO" in out
        assert "CFO" in out


class TestCmdFile:
    def test_execute_file(self, capsys):
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

    def test_file_not_found(self, capsys):
        rc = cmd_file("/nonexistent/path.ail")
        assert rc == 1
        err = capsys.readouterr().err
        assert "not found" in err.lower() or "cannot read" in err.lower()


class TestMain:
    def test_no_command_shows_help(self, capsys):
        rc = main([])
        assert rc == 0

    def test_parse_command(self, capsys):
        rc = main(["parse", "@CEO DECIDE expansion"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Directive" in out

    def test_emit_command(self, capsys):
        rc = main(["emit", "@CEO DECIDE expansion"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "@CEO" in out

    def test_validate_command(self, capsys):
        rc = main(["validate", "@CEO DECIDE expansion"])
        assert rc == 0

    def test_run_command(self, capsys):
        rc = main(["run", "@CEO DECIDE expansion"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "[echo]" in out


class TestFormatAst:
    def test_format_conditional(self):
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

    def test_format_parallel(self):
        node = ParallelBlockNode(branches=[
            DirectiveNode(
                action="FORECAST",
                recipient=RecipientNode(agent_code="CFO"),
            ),
            DirectiveNode(
                action="ASSESS",
                recipient=RecipientNode(agent_code="CTO"),
            ),
        ])
        text = _format_ast(node)
        assert "ParallelBlock" in text
        assert "CFO" in text
        assert "CTO" in text
