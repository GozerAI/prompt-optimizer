"""Caching infrastructure — hash-based memoization for compiled templates and optimization results.

Provides:
- TemplateCache: Pre-compilation cache for grammar templates (hash-based memoization)
- OptimizationCache: Result cache per input hash for full optimization results
- LRU eviction when cache exceeds max_size
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CacheEntry:
    """A cached entry with metadata."""

    key: str
    value: Any
    created_at: float = field(default_factory=time.monotonic)
    hit_count: int = 0
    size_estimate: int = 0


class LRUCache:
    """Thread-safe LRU cache with optional TTL eviction."""

    def __init__(self, max_size: int = 1024, ttl_seconds: float = 0.0) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Get a cached value. Returns None on miss."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if self._ttl > 0 and (time.monotonic() - entry.created_at) > self._ttl:
                del self._store[key]
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            self._store[key] = CacheEntry(
                key=entry.key,
                value=entry.value,
                created_at=entry.created_at,
                hit_count=entry.hit_count + 1,
                size_estimate=entry.size_estimate,
            )
            return entry.value

    def put(self, key: str, value: Any, size_estimate: int = 0) -> None:
        """Store a value in cache, evicting LRU entries if needed."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = CacheEntry(
                key=key,
                value=value,
                size_estimate=size_estimate,
            )
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def invalidate(self, key: str) -> bool:
        """Remove a specific key. Returns True if it existed."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries and reset stats."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


def _content_hash(text: str) -> str:
    """Produce a stable hash for text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TemplateCache:
    """Pre-compilation cache for grammar templates.

    Caches compiled AST nodes keyed by the hash of the input text.
    Avoids re-parsing identical grammar strings.
    """

    def __init__(self, max_size: int = 512) -> None:
        self._cache = LRUCache(max_size=max_size)

    def get_compiled(self, text: str) -> Any | None:
        """Retrieve a previously compiled AST for this text."""
        key = _content_hash(text)
        return self._cache.get(key)

    def store_compiled(self, text: str, ast_node: Any) -> None:
        """Store a compiled AST node for this text."""
        key = _content_hash(text)
        self._cache.put(key, ast_node, size_estimate=len(text))

    def clear(self) -> None:
        self._cache.clear()

    @property
    def stats(self) -> dict[str, Any]:
        return self._cache.stats


class OptimizationCache:
    """Result cache for optimization outputs.

    Caches CompressedPrompt results keyed by input hash + optimization parameters.
    """

    def __init__(self, max_size: int = 256, ttl_seconds: float = 300.0) -> None:
        self._cache = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)

    def _make_key(
        self,
        text: str,
        max_layer: int = 3,
        min_fidelity: float = 0.50,
        target_reduction: float | None = None,
    ) -> str:
        """Build a composite cache key from input + params."""
        parts = f"{text}|ml={max_layer}|mf={min_fidelity}|tr={target_reduction}"
        return _content_hash(parts)

    def get(
        self,
        text: str,
        max_layer: int = 3,
        min_fidelity: float = 0.50,
        target_reduction: float | None = None,
    ) -> Any | None:
        key = self._make_key(text, max_layer, min_fidelity, target_reduction)
        return self._cache.get(key)

    def store(
        self,
        text: str,
        result: Any,
        max_layer: int = 3,
        min_fidelity: float = 0.50,
        target_reduction: float | None = None,
    ) -> None:
        key = self._make_key(text, max_layer, min_fidelity, target_reduction)
        self._cache.put(key, result, size_estimate=len(text))

    def clear(self) -> None:
        self._cache.clear()

    @property
    def stats(self) -> dict[str, Any]:
        return self._cache.stats
