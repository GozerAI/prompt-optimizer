"""Response contract enforcement for directives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prompt_optimizer.grammar.ast_nodes import ResponseContractNode

# Mapping from type hints to Python types
_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
}


class ContractViolationError(ValueError):
    """Raised when a directive output violates its response contract."""

    def __init__(self, violation: ContractViolation) -> None:
        self.violation = violation
        super().__init__(violation.message)


@dataclass
class ContractViolation:
    """Details of a contract violation."""

    missing_fields: list[str] = field(default_factory=list)
    type_errors: list[tuple[str, str, str]] = field(default_factory=list)  # (field, expected, actual)
    message: str = ""

    def __post_init__(self) -> None:
        if not self.message:
            self.message = self._build_message()

    def _build_message(self) -> str:
        parts: list[str] = []
        if self.missing_fields:
            parts.append(f"Missing required fields: {', '.join(self.missing_fields)}")
        for name, expected, actual in self.type_errors:
            parts.append(f"Field '{name}': expected {expected}, got {actual}")
        return "; ".join(parts) if parts else "Contract violation"


class ContractEnforcer:
    """Validate directive output against a ResponseContractNode."""

    def validate(self, output: Any, contract: ResponseContractNode) -> ContractViolation | None:
        """Return None if valid, ContractViolation if not."""
        if not contract.fields:
            return None

        missing: list[str] = []
        type_errors: list[tuple[str, str, str]] = []

        # Output must be a dict to check fields
        if not isinstance(output, dict):
            return ContractViolation(
                missing_fields=[f.name for f in contract.fields if f.required],
                message=f"Expected dict output for contract validation, got {type(output).__name__}",
            )

        for cf in contract.fields:
            if cf.name not in output:
                if cf.required:
                    missing.append(cf.name)
                continue

            # Type check (skip for "any")
            if cf.type_hint.lower() == "any":
                continue

            expected_type = _TYPE_MAP.get(cf.type_hint.lower())
            if expected_type is None:
                # Unknown type hint — skip validation
                continue

            actual = output[cf.name]
            if not isinstance(actual, expected_type):
                # Allow int where float is expected
                if expected_type is float and isinstance(actual, int):
                    continue
                type_errors.append((cf.name, cf.type_hint, type(actual).__name__))

        if missing or type_errors:
            return ContractViolation(missing_fields=missing, type_errors=type_errors)

        return None
