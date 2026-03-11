"""Lexer for the agent communication grammar.

Tokenizes compact agent notation like:
    @CFO ANALYZE revenue {period=Q1-2026} -> summary !urgent ~thorough
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
    "APPROVE", "COST",
}


class LexError(Exception):
    """Lexer error with position information."""

    def __init__(self, message: str, position: int, line: int = 1):
        self.position = position
        self.line = line
        super().__init__(f"Lex error at pos {position}: {message}")


class Lexer:
    """Tokenizer for the agent communication grammar."""

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

            # Single-character tokens
            single_map = {
                "@": TokenType.AT,
                "|": TokenType.PIPE,
                "{": TokenType.LBRACE,
                "}": TokenType.RBRACE,
                "[": TokenType.LBRACKET,
                "]": TokenType.RBRACKET,
                "!": TokenType.BANG,
                "~": TokenType.TILDE,
                ",": TokenType.COMMA,
                "=": TokenType.EQUALS,
                ":": TokenType.COLON,
                ">": TokenType.GT,
                "<": TokenType.LT,
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

            # Number
            if ch.isdigit() or (ch == "-" and pos + 1 < len(text) and text[pos + 1].isdigit()):
                match = re.match(r"-?[\d,.]+%?", text[pos:])
                if match:
                    tokens.append(Token(TokenType.NUMBER, match.group(), pos, line))
                    pos += match.end()
                    continue

            # Word (identifier, keyword, agent code, or action)
            if ch.isalpha() or ch == "_":
                match = re.match(r"[A-Za-z_][\w\-]*", text[pos:])
                if match:
                    word = match.group()
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

                    pos += match.end()
                    continue

            # Dollar sign (part of number like $2.3M)
            if ch == "$":
                match = re.match(r"\$[\d,.]+[MBKmk]?", text[pos:])
                if match:
                    tokens.append(Token(TokenType.NUMBER, match.group(), pos, line))
                    pos += match.end()
                    continue

            # Skip unknown characters
            pos += 1

        tokens.append(Token(TokenType.EOF, "", pos, line))
        return tokens
