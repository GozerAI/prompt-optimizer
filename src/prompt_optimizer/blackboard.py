"""Blackboard — shared state store for context pointers."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class VersionedEntry:
    """A versioned entry in the blackboard."""

    namespace: str
    key: str
    value: Any
    version: int
    content_hash: str
    timestamp: float = field(default_factory=time.time)

    @property
    def pointer(self) -> str:
        """Return a pointer string like 'org:system_state@v3'."""
        return f"{self.namespace}:{self.key}@v{self.version}"


class Blackboard:
    """Shared state store for context pointers used by Layer 3 compression."""

    def __init__(self) -> None:
        self._store: dict[str, list[VersionedEntry]] = {}

    def _hash(self, value: Any) -> str:
        return hashlib.sha256(str(value).encode()).hexdigest()[:12]

    def _make_key(self, namespace: str, key: str) -> str:
        return f"{namespace}:{key}"

    def put(self, namespace: str, key: str, value: Any) -> str:
        """Store a value, return versioned pointer."""
        store_key = self._make_key(namespace, key)
        content_hash = self._hash(value)

        # Check if value unchanged
        if store_key in self._store:
            latest = self._store[store_key][-1]
            if latest.content_hash == content_hash:
                return latest.pointer
            version = latest.version + 1
        else:
            self._store[store_key] = []
            version = 1

        entry = VersionedEntry(
            namespace=namespace,
            key=key,
            value=value,
            version=version,
            content_hash=content_hash,
        )
        self._store[store_key].append(entry)
        return entry.pointer

    def get(self, pointer: str) -> Any:
        """Resolve a pointer to its value.

        Pointer format: 'namespace:key@vN' or 'namespace:key' (latest).
        """
        if "@v" in pointer:
            base, version_str = pointer.rsplit("@v", 1)
            version = int(version_str)
        else:
            base = pointer
            version = None

        if base not in self._store:
            raise KeyError(f"Blackboard key not found: {base}")

        entries = self._store[base]
        if version is not None:
            for entry in entries:
                if entry.version == version:
                    return entry.value
            raise KeyError(f"Version {version} not found for {base}")

        return entries[-1].value

    def has(self, pointer: str) -> bool:
        """Check if a pointer exists."""
        try:
            self.get(pointer)
            return True
        except KeyError:
            return False

    def get_latest_pointer(self, namespace: str, key: str) -> Optional[str]:
        """Get the latest pointer for a namespace:key pair."""
        store_key = self._make_key(namespace, key)
        if store_key not in self._store:
            return None
        return self._store[store_key][-1].pointer

    def snapshot(self) -> dict[str, Any]:
        """Full state for serialization/sync."""
        result: dict[str, Any] = {}
        for store_key, entries in self._store.items():
            latest = entries[-1]
            result[latest.pointer] = latest.value
        return result

    def diff(self, other: Blackboard) -> list[str]:
        """Find divergent entries between two blackboard instances."""
        divergent: list[str] = []

        all_keys = set(self._store.keys()) | set(other._store.keys())
        for key in all_keys:
            self_entries = self._store.get(key, [])
            other_entries = other._store.get(key, [])

            if not self_entries or not other_entries:
                divergent.append(key)
            elif self_entries[-1].content_hash != other_entries[-1].content_hash:
                divergent.append(key)

        return divergent

    def clear(self) -> None:
        """Clear all entries."""
        self._store.clear()
