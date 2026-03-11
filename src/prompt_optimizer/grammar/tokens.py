"""Token types and Token dataclass for the agent communication grammar."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    # Delimiters
    AT = auto()          # @
    PIPE = auto()        # |
    LBRACE = auto()      # {
    RBRACE = auto()      # }
    LBRACKET = auto()    # [
    RBRACKET = auto()    # ]
    BANG = auto()        # !
    TILDE = auto()       # ~
    ARROW = auto()       # ->
    COMMA = auto()       # ,
    EQUALS = auto()      # =
    COLON = auto()       # :

    # Keywords
    IF = auto()
    THEN = auto()
    ELSE = auto()

    # Semantic tokens
    AGENT_CODE = auto()  # CEO, CFO, CTO, etc.
    ACTION = auto()      # ANALYZE, ASSESS, DECIDE, etc.
    IDENTIFIER = auto()  # general words
    STRING = auto()      # quoted "strings"
    NUMBER = auto()      # numeric values

    # Comparators
    GT = auto()          # >
    LT = auto()          # <
    GTE = auto()         # >=
    LTE = auto()         # <=
    EQ = auto()          # ==
    NEQ = auto()         # !=

    # Structure
    EOF = auto()
    NEWLINE = auto()


# Sets for classification
COMPARATORS = {TokenType.GT, TokenType.LT, TokenType.GTE, TokenType.LTE, TokenType.EQ, TokenType.NEQ, TokenType.COLON}

KEYWORDS = {"IF": TokenType.IF, "THEN": TokenType.THEN, "ELSE": TokenType.ELSE}


@dataclass(frozen=True)
class Token:
    """A single token from the lexer."""

    type: TokenType
    value: str
    position: int
    line: int = 1

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, pos={self.position})"
