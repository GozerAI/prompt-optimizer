"""Lexer for the agent communication grammar.

Tokenizes compact agent notation like:
    @CFO ANALYZE revenue {period=Q1-2026} -> summary !urgent ~thorough
    PAR { @CFO FORECAST revenue; @CTO ASSESS infra }
    @CDO GATHER data | @CFO ANALYZE $prev
"""

from __future__ import annotations

import re

from prompt_optimizer.grammar.tokens import KEYWORDS, Token, TokenType

# Default known agent codes
DEFAULT_AGENT_CODES = {
    "CEO", "COO", "CTO", "CFO", "CIO", "CMO", "CHRO", "CLO",
    "CSO", "CDO", "CPO", "CRO", "CSTRO", "CINO", "CRISKO", "CSUSO",
    "CCO", "CSecO", "CComO", "CEngO", "CRiO", "CRevO", "CCoMO",
}

# Default known action verbs (uppercase)
DEFAULT_ACTIONS = {
    "ANALYZE", "ASSESS", "EVALUATE", "REVIEW", "GENERATE", "CREATE",
    "DECIDE", "DETERMINE", "SUMMARIZE", "DELEGATE", "RECOMMEND",
    "FORECAST", "ESTIMATE", "REPORT", "PLAN", "MONITOR", "OPTIMIZE",
    "APPROVE", "COST", "GATHER", "EXECUTE", "UPDATE", "DELETE",
    "QUERY", "NOTIFY", "VALIDATE",
}

# Priority levels recognized after !
_PRIORITY_LEVELS = {"urgent", "high", "normal", "low"}


class LexError(Exception):
    """Lexer error with position information."""

    def __init__(self, message: str, position: int, line: int = 1):
        self.position = position
        self.line = line
        super().__init__(f"Lex error at pos {position}: {message}")


class Lexer:
    """Tokenizer for the agent communication grammar.

    Supports both the original prompt-optimizer notation and AIL extensions
    (PAR/SEQ blocks, $prev refs, bb: refs, RETRY, response contracts).
    """

    def __init__(
        self,
        agent_codes: set[str] | None = None,
        actions: set[str] | None = None,
    ) -> None:
        self._agent_codes = agent_codes or DEFAULT_AGENT_CODES
        self._actions = actions or DEFAULT_ACTIONS

    def tokenize(self, text: str) -> list[Token]:
        """Tokenize input text into a list of tokens."""
        tokens: list[Token] = []
        pos = 0
        line = 1

        while pos < len(text):
            ch = text[pos]

            # Skip whitespace (except newlines)
            if ch in " \t\r":
                pos += 1
                continue

            # Comments: # to end of line
            if ch == "#":
                end = text.find("\n", pos)
                if end == -1:
                    break
                pos = end
                continue

            # Newline
            if ch == "\n":
                tokens.append(Token(TokenType.NEWLINE, "\n", pos, line))
                line += 1
                pos += 1
                continue

            # Two-character tokens
            if pos + 1 < len(text):
                two = text[pos:pos + 2]
                if two == "->":
                    tokens.append(Token(TokenType.ARROW, "->", pos, line))
                    pos += 2
                    continue
                if two == ">=":
                    tokens.append(Token(TokenType.GTE, ">=", pos, line))
                    pos += 2
                    continue
                if two == "<=":
                    tokens.append(Token(TokenType.LTE, "<=", pos, line))
                    pos += 2
                    continue
                if two == "==":
                    tokens.append(Token(TokenType.EQ, "==", pos, line))
                    pos += 2
                    continue
                if two == "!=":
                    tokens.append(Token(TokenType.NEQ, "!=", pos, line))
                    pos += 2
                    continue

            # $prev / $prev[N]
            if ch == "$":
                m = re.match(r"\$prev(?:\[(\d+)\])?", text[pos:])
                if m:
                    tokens.append(Token(TokenType.PREV_REF, m.group(), pos, line))
                    pos += m.end()
                    continue
                # Dollar amount: $2.3M
                m = re.match(r"\$[\d,.]+[MBKmk]?", text[pos:])
                if m:
                    tokens.append(Token(TokenType.NUMBER, m.group(), pos, line))
                    pos += m.end()
                    continue

            # bb:namespace:key@vN
            if text[pos:pos + 3] == "bb:":
                m = re.match(
                    r"bb:([a-z_][a-z0-9_]*):([a-z_][a-z0-9_]*)(?:@v(\d+))?",
                    text[pos:],
                )
                if m:
                    tokens.append(Token(TokenType.BB_REF, m.group(), pos, line))
                    pos += m.end()
                    continue

            # @ — agent ref or agent field ref
            if ch == "@":
                # @AGENT.field
                m = re.match(r"@([A-Z][A-Za-z0-9_]*)\.([a-z_][a-z0-9_]*)", text[pos:])
                if m:
                    tokens.append(Token(TokenType.AGENT_FIELD, m.group(), pos, line))
                    pos += m.end()
                    continue
                # @AGENT — emit AT + AGENT_CODE separately for backward compat
                m = re.match(r"@([A-Z][A-Za-z0-9_]*)", text[pos:])
                if m:
                    tokens.append(Token(TokenType.AT, "@", pos, line))
                    code = m.group(1)
                    tokens.append(Token(TokenType.AGENT_CODE, code, pos + 1, line))
                    pos += m.end()
                    continue
                # Just @ by itself
                tokens.append(Token(TokenType.AT, "@", pos, line))
                pos += 1
                continue

            # ! — priority (!urgent, !high, !low, !normal) or just bang
            if ch == "!":
                m = re.match(r"!(urgent|high|normal|low)", text[pos:])
                if m:
                    tokens.append(Token(TokenType.PRIORITY, m.group(1), pos, line))
                    pos += m.end()
                    continue
                tokens.append(Token(TokenType.BANG, "!", pos, line))
                pos += 1
                continue

            # ~ — modifier
            if ch == "~":
                m = re.match(r"~([a-z]+)", text[pos:])
                if m:
                    tokens.append(Token(TokenType.MODIFIER, m.group(1), pos, line))
                    pos += m.end()
                    continue
                tokens.append(Token(TokenType.TILDE, "~", pos, line))
                pos += 1
                continue

            # Single-character tokens
            single_map = {
                "|": TokenType.PIPE,
                "{": TokenType.LBRACE,
                "}": TokenType.RBRACE,
                "[": TokenType.LBRACKET,
                "]": TokenType.RBRACKET,
                ",": TokenType.COMMA,
                "=": TokenType.EQUALS,
                ":": TokenType.COLON,
                ">": TokenType.GT,
                "<": TokenType.LT,
                ";": TokenType.SEMICOLON,
                "?": TokenType.QUESTION,
            }

            if ch in single_map:
                tokens.append(Token(single_map[ch], ch, pos, line))
                pos += 1
                continue

            # Quoted string
            if ch in ('"', "'"):
                end = text.find(ch, pos + 1)
                if end == -1:
                    end = len(text)
                value = text[pos + 1:end]
                tokens.append(Token(TokenType.STRING, value, pos, line))
                pos = end + 1
                continue

            # Number (including suffixed like 1M, 100k, percentages)
            if ch.isdigit() or (ch == "-" and pos + 1 < len(text) and text[pos + 1].isdigit()):
                m = re.match(r"-?\d(?:,\d{3}|\d)*(?:\.\d+)?[MkBT%]?", text[pos:])
                if m:
                    tokens.append(Token(TokenType.NUMBER, m.group(), pos, line))
                    pos += m.end()
                    continue

            # Word (identifier, keyword, agent code, or action)
            if ch.isalpha() or ch == "_":
                m = re.match(r"[A-Za-z_][\w\-]*", text[pos:])
                if m:
                    word = m.group()
                    upper = word.upper()

                    # Check keywords first
                    if upper in KEYWORDS:
                        tokens.append(Token(KEYWORDS[upper], word, pos, line))
                    # Check agent codes
                    elif upper in self._agent_codes or word in self._agent_codes:
                        tokens.append(Token(TokenType.AGENT_CODE, word, pos, line))
                    # Check actions
                    elif upper in self._actions:
                        tokens.append(Token(TokenType.ACTION, upper, pos, line))
                    else:
                        tokens.append(Token(TokenType.IDENTIFIER, word, pos, line))

                    pos += m.end()
                    continue

            # Skip unknown characters
            pos += 1

        tokens.append(Token(TokenType.EOF, "", pos, line))
        return tokens
