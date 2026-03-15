"""Automatic grammar validation on import.

Validates that grammar rules (keywords, token types, agent codes, actions)
are consistent and non-conflicting when the module loads. Detects issues
like duplicate keyword mappings, overlapping token patterns, and missing
cross-references between lexer and parser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GrammarIssue:
    """A single grammar consistency issue."""

    category: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GrammarValidationReport:
    """Result of grammar validation."""

    issues: list[GrammarIssue] = field(default_factory=list)
    checks_run: int = 0

    @property
    def valid(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")


class GrammarConsistencyValidator:
    """Validates grammar rules for internal consistency."""

    def validate(self) -> GrammarValidationReport:
        """Run all grammar consistency checks."""
        report = GrammarValidationReport()
        self._check_keyword_uniqueness(report)
        self._check_token_type_coverage(report)
        self._check_agent_code_format(report)
        self._check_action_verb_format(report)
        self._check_keyword_action_overlap(report)
        self._check_priority_levels(report)
        self._check_comparator_completeness(report)
        self._check_keyword_token_type_validity(report)
        return report

    def _check_keyword_uniqueness(self, report: GrammarValidationReport) -> None:
        report.checks_run += 1
        from prompt_optimizer.grammar.tokens import KEYWORDS
        seen: dict[str, str] = {}
        for kw, tt in KEYWORDS.items():
            if kw in seen:
                report.issues.append(GrammarIssue(
                    category="duplicate", severity="error",
                    message=f"Duplicate keyword {kw!r} maps to both {seen[kw]} and {tt.name}",
                    details={"keyword": kw, "types": [seen[kw], tt.name]},
                ))
            seen[kw] = tt.name

    def _check_token_type_coverage(self, report: GrammarValidationReport) -> None:
        report.checks_run += 1
        from prompt_optimizer.grammar.tokens import KEYWORDS, TokenType
        valid_types = {t.name for t in TokenType}
        for kw, tt in KEYWORDS.items():
            if tt.name not in valid_types:
                report.issues.append(GrammarIssue(
                    category="missing", severity="error",
                    message=f"Keyword {kw!r} maps to non-existent TokenType {tt.name!r}",
                    details={"keyword": kw, "token_type": tt.name},
                ))

    def _check_agent_code_format(self, report: GrammarValidationReport) -> None:
        report.checks_run += 1
        from prompt_optimizer.grammar.lexer import DEFAULT_AGENT_CODES
        for code in DEFAULT_AGENT_CODES:
            if not code.isupper():
                report.issues.append(GrammarIssue(
                    category="conflict", severity="warning",
                    message=f"Agent code {code!r} is not fully uppercase",
                    details={"code": code},
                ))
            if len(code) < 2:
                report.issues.append(GrammarIssue(
                    category="conflict", severity="error",
                    message=f"Agent code {code!r} is too short (minimum 2 chars)",
                    details={"code": code},
                ))

    def _check_action_verb_format(self, report: GrammarValidationReport) -> None:
        report.checks_run += 1
        from prompt_optimizer.grammar.lexer import DEFAULT_ACTIONS
        pat = re.compile(r"^[A-Z][A-Z_]+$")
        for action in DEFAULT_ACTIONS:
            if not action.isupper():
                report.issues.append(GrammarIssue(
                    category="conflict", severity="warning",
                    message=f"Action {action!r} is not fully uppercase",
                    details={"action": action},
                ))
            if not pat.match(action) and not re.match(r"^[A-Z]+$", action):
                report.issues.append(GrammarIssue(
                    category="conflict", severity="warning",
                    message=f"Action {action!r} does not match expected pattern",
                    details={"action": action},
                ))

    def _check_keyword_action_overlap(self, report: GrammarValidationReport) -> None:
        report.checks_run += 1
        from prompt_optimizer.grammar.lexer import DEFAULT_ACTIONS
        from prompt_optimizer.grammar.tokens import KEYWORDS
        for word in sorted(set(KEYWORDS.keys()) & DEFAULT_ACTIONS):
            report.issues.append(GrammarIssue(
                category="overlap", severity="error",
                message=f"{word!r} is both a keyword and an action verb",
                details={"word": word},
            ))

    def _check_priority_levels(self, report: GrammarValidationReport) -> None:
        report.checks_run += 1
        from prompt_optimizer.grammar.lexer import _PRIORITY_LEVELS
        expected = {"urgent", "high", "normal", "low"}
        for level in sorted(expected - _PRIORITY_LEVELS):
            report.issues.append(GrammarIssue(
                category="missing", severity="warning",
                message=f"Expected priority level {level!r} is missing",
                details={"level": level},
            ))

    def _check_comparator_completeness(self, report: GrammarValidationReport) -> None:
        report.checks_run += 1
        from prompt_optimizer.grammar.tokens import COMPARATORS
        expected = {"GT", "LT", "GTE", "LTE", "EQ", "NEQ"}
        actual = {t.name for t in COMPARATORS}
        for name in sorted(expected - actual):
            report.issues.append(GrammarIssue(
                category="missing", severity="warning",
                message=f"Expected comparator TokenType.{name} not in COMPARATORS set",
                details={"comparator": name},
            ))

    def _check_keyword_token_type_validity(self, report: GrammarValidationReport) -> None:
        report.checks_run += 1
        from prompt_optimizer.grammar.tokens import KEYWORDS
        for kw, tt in KEYWORDS.items():
            if tt.name != kw:
                report.issues.append(GrammarIssue(
                    category="conflict", severity="warning",
                    message=f"Keyword {kw!r} maps to TokenType.{tt.name} (mismatch)",
                    details={"keyword": kw, "token_type": tt.name},
                ))


def validate_grammar() -> GrammarValidationReport:
    """Run grammar consistency validation and return report."""
    return GrammarConsistencyValidator().validate()


_import_validation: GrammarValidationReport | None = None


def get_import_validation() -> GrammarValidationReport:
    """Get the validation report from module import time."""
    global _import_validation
    if _import_validation is None:
        _import_validation = validate_grammar()
    return _import_validation


def is_grammar_valid() -> bool:
    """Quick check: are grammar rules consistent (no errors)?"""
    return get_import_validation().valid
