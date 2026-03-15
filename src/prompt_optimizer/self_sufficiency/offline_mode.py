"""Offline operation mode for prompt optimization.

Provides token counting and optimization capabilities without external
dependencies. Uses the built-in memory_tokenizer when tiktoken is unavailable,
with automatic availability detection.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable


class TokenizerBackend(Enum):
    """Available tokenizer backends."""

    TIKTOKEN = auto()
    MEMORY = auto()


@dataclass
class BackendInfo:
    """Information about a tokenizer backend."""

    backend: TokenizerBackend
    name: str
    available: bool
    accuracy: str
    description: str


@dataclass
class OfflineModeStatus:
    """Current offline mode status."""

    offline: bool
    active_backend: TokenizerBackend
    backends: list[BackendInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _probe_tiktoken() -> bool:
    """Check if tiktoken is importable and functional."""
    try:
        tiktoken = importlib.import_module("tiktoken")
        tiktoken.get_encoding("cl100k_base")
        return True
    except (ImportError, Exception):
        return False


def _probe_memory_tokenizer() -> bool:
    """Check if the built-in memory tokenizer is available."""
    try:
        from prompt_optimizer.memory_tokenizer import count_tokens_efficient
        return callable(count_tokens_efficient)
    except ImportError:
        return False


class OfflineTokenizer:
    """Token counter that auto-selects the best available backend.

    Prefers tiktoken for accuracy, falls back to memory_tokenizer for
    zero-dep operation. Can be forced to a specific backend.
    """

    def __init__(self, force_backend: TokenizerBackend | None = None) -> None:
        self._tiktoken_available = _probe_tiktoken()
        self._memory_available = _probe_memory_tokenizer()
        self._force_backend = force_backend
        self._backend = self._select_backend()
        self._count_fn: Callable[[str], int] = self._build_count_fn()

    def _select_backend(self) -> TokenizerBackend:
        """Select the best available backend."""
        if self._force_backend is not None:
            if self._force_backend == TokenizerBackend.TIKTOKEN and not self._tiktoken_available:
                raise RuntimeError(
                    "tiktoken backend requested but not available. "
                    "Install with: pip install tiktoken"
                )
            if self._force_backend == TokenizerBackend.MEMORY and not self._memory_available:
                raise RuntimeError("memory_tokenizer backend requested but not available.")
            return self._force_backend
        if self._tiktoken_available:
            return TokenizerBackend.TIKTOKEN
        if self._memory_available:
            return TokenizerBackend.MEMORY
        raise RuntimeError("No tokenizer backend available.")

    def _build_count_fn(self) -> Callable[[str], int]:
        """Build the token counting function for the selected backend."""
        if self._backend == TokenizerBackend.TIKTOKEN:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return lambda text: len(enc.encode(text)) if text else 0
        from prompt_optimizer.memory_tokenizer import count_tokens_efficient
        return count_tokens_efficient

    @property
    def backend(self) -> TokenizerBackend:
        """The active tokenizer backend."""
        return self._backend

    @property
    def is_offline(self) -> bool:
        """True if running without tiktoken (offline mode)."""
        return self._backend != TokenizerBackend.TIKTOKEN

    @property
    def is_exact(self) -> bool:
        """True if using an exact tokenizer (tiktoken)."""
        return self._backend == TokenizerBackend.TIKTOKEN

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the active backend."""
        if not text:
            return 0
        return self._count_fn(text)

    def count_tokens_batch(self, texts: list[str]) -> list[int]:
        """Count tokens for multiple texts."""
        return [self.count_tokens(t) for t in texts]

    def estimate_reduction(self, original: str, compressed: str) -> float:
        """Estimate token reduction percentage."""
        orig_tokens = self.count_tokens(original)
        if orig_tokens == 0:
            return 0.0
        comp_tokens = self.count_tokens(compressed)
        return 1.0 - (comp_tokens / orig_tokens)

    def status(self) -> OfflineModeStatus:
        """Get current offline mode status."""
        backends = [
            BackendInfo(
                backend=TokenizerBackend.TIKTOKEN, name="tiktoken",
                available=self._tiktoken_available, accuracy="exact",
                description="OpenAI tiktoken cl100k_base encoding",
            ),
            BackendInfo(
                backend=TokenizerBackend.MEMORY, name="memory_tokenizer",
                available=self._memory_available, accuracy="approximate",
                description="Built-in BPE estimation (~5% accuracy)",
            ),
        ]
        warnings = []
        if self.is_offline:
            warnings.append(
                "Running in offline mode with approximate token counts. "
                "Install tiktoken for exact counts: pip install tiktoken"
            )
        return OfflineModeStatus(
            offline=self.is_offline, active_backend=self._backend,
            backends=backends, warnings=warnings,
        )


_default_tokenizer: OfflineTokenizer | None = None


def get_offline_tokenizer() -> OfflineTokenizer:
    """Get or create the default OfflineTokenizer singleton."""
    global _default_tokenizer
    if _default_tokenizer is None:
        _default_tokenizer = OfflineTokenizer()
    return _default_tokenizer


def reset_offline_tokenizer() -> None:
    """Reset the singleton (useful for testing)."""
    global _default_tokenizer
    _default_tokenizer = None


def count_tokens_offline(text: str) -> int:
    """Count tokens using the best available backend."""
    return get_offline_tokenizer().count_tokens(text)


def is_offline() -> bool:
    """Check if running in offline mode (no tiktoken)."""
    return get_offline_tokenizer().is_offline
