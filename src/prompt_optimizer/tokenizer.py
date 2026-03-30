"""Token counting with optional tiktoken support."""

from __future__ import annotations


def count_tokens(text: str) -> int:
    """Count tokens in text. Uses tiktoken if available, else word-split heuristic."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Heuristic: ~1.3 tokens per whitespace-delimited word
        words = text.split()
        return max(1, int(len(words) * 1.3))
