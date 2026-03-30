"""Blackboard — shared state store for context pointers.

Provides versioned, namespaced shared state for inter-agent context.
Supports subscriptions for change notifications and staleness detection.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Subscriber callback type: (namespace, key, value) -> None
SubscriberCallback = Callable[[str, str, Any], None]


@dataclass
class VersionedEntry:
    """A versioned entry in the blackboard."""

    namespace: str
    key: str
    value: Any
    version: int
    content_hash: str
    timestamp: float = field(default_factory=time.time)
    source_agent: Optional[str] = None

    @property
    def pointer(self) -> str:
        """Return a pointer string like 'org:system_state@v3'."""
        return f"{self.namespace}:{self.key}@v{self.version}"

    @property
    def age_seconds(self) -> float:
        """Seconds since this entry was last updated."""
        return time.time() - self.timestamp


class Blackboard:
    """Shared state store for context pointers.

    Features:
    - Versioned entries with content hashing (no-op on unchanged data)
    - Namespace-based subscriptions for change notifications
    - Staleness detection for outdated context
    - Snapshot/restore for session management
    """

    def __init__(self, staleness_threshold: float = 3600.0) -> None:
        self._store: dict[str, list[VersionedEntry]] = {}
        self._subscribers: dict[str, list[SubscriberCallback]] = {}
        self.staleness_threshold = staleness_threshold

    def _hash(self, value: Any) -> str:
        return hashlib.sha256(str(value).encode()).hexdigest()[:12]

    def _make_key(self, namespace: str, key: str) -> str:
        return f"{namespace}:{key}"

    def put(
        self, namespace: str, key: str, value: Any, source_agent: str | None = None
    ) -> str:
        """Store a value, return versioned pointer. Notifies subscribers."""
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
            source_agent=source_agent,
        )
        self._store[store_key].append(entry)

        # Notify subscribers
        self._notify(namespace, key, value)

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

    # --- Subscriptions ---

    def subscribe(self, namespace: str, callback: SubscriberCallback) -> None:
        """Subscribe to changes in a namespace."""
        if namespace not in self._subscribers:
            self._subscribers[namespace] = []
        self._subscribers[namespace].append(callback)

    def unsubscribe(self, namespace: str, callback: SubscriberCallback) -> None:
        """Unsubscribe from a namespace."""
        if namespace in self._subscribers:
            self._subscribers[namespace] = [
                cb for cb in self._subscribers[namespace] if cb is not callback
            ]

    def _notify(self, namespace: str, key: str, value: Any) -> None:
        """Notify subscribers of a change."""
        for callback in self._subscribers.get(namespace, []):
            try:
                callback(namespace, key, value)
            except Exception:
                pass  # Subscribers should not break the blackboard

    # --- Staleness ---

    def get_stale(self, threshold: float | None = None) -> list[str]:
        """Return pointers whose latest entry is older than threshold seconds."""
        threshold = threshold or self.staleness_threshold
        now = time.time()
        stale: list[str] = []

        for entries in self._store.values():
            latest = entries[-1]
            if (now - latest.timestamp) > threshold:
                stale.append(latest.pointer)

        return stale

    def is_stale(self, pointer: str, threshold: float | None = None) -> bool:
        """Check if a specific pointer is stale."""
        threshold = threshold or self.staleness_threshold
        try:
            base = pointer.rsplit("@v", 1)[0] if "@v" in pointer else pointer
            if base not in self._store:
                return True
            latest = self._store[base][-1]
            return latest.age_seconds > threshold
        except (KeyError, IndexError):
            return True

    # --- Namespaces ---

    @property
    def namespaces(self) -> list[str]:
        """List all namespaces that have entries."""
        ns: set[str] = set()
        for entries in self._store.values():
            ns.add(entries[-1].namespace)
        return sorted(ns)

    # --- Restore ---

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore state from a snapshot (inverse of snapshot())."""
        self._store.clear()
        for pointer, value in snapshot.items():
            # Parse pointer: namespace:key@vN
            if "@v" in pointer:
                base, _ = pointer.rsplit("@v", 1)
            else:
                base = pointer
            parts = base.split(":", 1)
            if len(parts) == 2:
                self.put(parts[0], parts[1], value)
