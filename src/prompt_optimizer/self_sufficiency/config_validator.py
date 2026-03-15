"""Configuration self-validation at startup.

Validates all optimizer settings against schema, warns on invalid
combinations. Provides a declarative schema for configuration parameters
with type checking, range validation, and cross-field constraint enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable


class ParamType(Enum):
    """Configuration parameter types."""
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    BOOL = auto()
    LIST = auto()


@dataclass
class ParamSchema:
    """Schema definition for a single configuration parameter."""
    name: str
    param_type: ParamType
    description: str
    default: Any = None
    required: bool = False
    min_value: float | int | None = None
    max_value: float | int | None = None
    allowed_values: list[Any] | None = None
    item_type: ParamType | None = None


@dataclass
class ConfigIssue:
    """A single configuration validation issue."""
    param: str
    severity: str
    message: str


@dataclass
class ConfigValidationReport:
    """Result of configuration validation."""
    issues: list[ConfigIssue] = field(default_factory=list)
    checked_params: int = 0

    @property
    def valid(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")


CrossFieldConstraint = Callable[[dict[str, Any]], list[ConfigIssue]]


OPTIMIZER_SCHEMA: list[ParamSchema] = [
    ParamSchema(
        name="min_fidelity", param_type=ParamType.FLOAT,
        description="Minimum fidelity score before stopping optimization",
        default=0.50, min_value=0.0, max_value=1.0,
    ),
    ParamSchema(
        name="max_risk", param_type=ParamType.FLOAT,
        description="Maximum risk score allowed per layer",
        default=0.25, min_value=0.0, max_value=1.0,
    ),
    ParamSchema(
        name="max_layer", param_type=ParamType.INT,
        description="Maximum compression layer to apply (1-3)",
        default=3, min_value=1, max_value=3,
    ),
    ParamSchema(
        name="target_reduction", param_type=ParamType.FLOAT,
        description="Target token reduction percentage (0.0-1.0)",
        default=None, required=False, min_value=0.0, max_value=1.0,
    ),
    ParamSchema(
        name="chunk_size", param_type=ParamType.INT,
        description="Chunk size for memory-efficient token counting",
        default=4096, min_value=64, max_value=1048576,
    ),
    ParamSchema(
        name="agent_codes", param_type=ParamType.LIST,
        description="List of recognized agent codes for grammar parsing",
        default=None, required=False, item_type=ParamType.STRING,
    ),
    ParamSchema(
        name="conversation_id", param_type=ParamType.STRING,
        description="Conversation ID for context tracking",
        default="", required=False,
    ),
]

def _constraint_fidelity_risk_balance(config: dict[str, Any]) -> list[ConfigIssue]:
    """Warn if min_fidelity is very low and max_risk is very high."""
    issues: list[ConfigIssue] = []
    mf = config.get("min_fidelity")
    mr = config.get("max_risk")
    if isinstance(mf, (int, float)) and isinstance(mr, (int, float)):
        if mf < 0.3 and mr > 0.5:
            issues.append(ConfigIssue(
                param="min_fidelity+max_risk", severity="warning",
                message=f"Low fidelity ({mf}) with high risk ({mr}) may produce unreliable output",
            ))
    return issues


def _constraint_layer_risk_coherence(config: dict[str, Any]) -> list[ConfigIssue]:
    """Warn if max_layer is 3 but max_risk is very low."""
    issues: list[ConfigIssue] = []
    ml = config.get("max_layer")
    mr = config.get("max_risk")
    if isinstance(ml, int) and isinstance(mr, (int, float)):
        if ml == 3 and mr < 0.10:
            issues.append(ConfigIssue(
                param="max_layer+max_risk", severity="warning",
                message=f"Layer 3 typically has risk >= 10%, but max_risk is {mr}. Layer 3 will likely be skipped.",
            ))
    return issues


def _constraint_reduction_vs_layer(config: dict[str, Any]) -> list[ConfigIssue]:
    """Warn if target_reduction is >80% but max_layer is 1."""
    issues: list[ConfigIssue] = []
    target = config.get("target_reduction")
    ml = config.get("max_layer")
    if isinstance(target, (int, float)) and isinstance(ml, int):
        if target > 0.80 and ml == 1:
            issues.append(ConfigIssue(
                param="target_reduction+max_layer", severity="warning",
                message=f"Target {target*100:.0f}% unlikely with only Layer 1. Typical L1 is 55-70%.",
            ))
    return issues


DEFAULT_CONSTRAINTS: list[CrossFieldConstraint] = [
    _constraint_fidelity_risk_balance,
    _constraint_layer_risk_coherence,
    _constraint_reduction_vs_layer,
]

class ConfigValidator:
    """Validates optimizer configuration against schema and constraints."""

    def __init__(
        self,
        schema: list[ParamSchema] | None = None,
        constraints: list[CrossFieldConstraint] | None = None,
    ) -> None:
        self._schema = schema or OPTIMIZER_SCHEMA
        self._constraints = constraints if constraints is not None else DEFAULT_CONSTRAINTS
        self._schema_map = {p.name: p for p in self._schema}

    @property
    def schema(self) -> list[ParamSchema]:
        """The parameter schema being validated against."""
        return list(self._schema)

    def validate(self, config: dict[str, Any]) -> ConfigValidationReport:
        """Validate a configuration dictionary against the schema."""
        report = ConfigValidationReport()
        for param in self._schema:
            report.checked_params += 1
            value = config.get(param.name)
            if param.required and value is None:
                report.issues.append(ConfigIssue(
                    param=param.name, severity="error",
                    message=f"Required parameter {param.name!r} is missing",
                ))
                continue
            if value is None:
                continue
            self._check_type(param, value, report)
            self._check_range(param, value, report)
            self._check_allowed(param, value, report)
        known = {p.name for p in self._schema}
        for key in config:
            if key not in known:
                report.checked_params += 1
                report.issues.append(ConfigIssue(
                    param=key, severity="warning",
                    message=f"Unknown parameter {key!r} is not in the schema",
                ))
        for constraint in self._constraints:
            report.issues.extend(constraint(config))
        return report

    def _check_type(self, param: ParamSchema, value: Any, report: ConfigValidationReport) -> None:
        """Check parameter type matches schema."""
        expected = {
            ParamType.INT: (int,),
            ParamType.FLOAT: (int, float),
            ParamType.STRING: (str,),
            ParamType.BOOL: (bool,),
            ParamType.LIST: (list, tuple),
        }
        valid_types = expected.get(param.param_type, ())
        if not isinstance(value, valid_types):
            report.issues.append(ConfigIssue(
                param=param.name, severity="error",
                message=f"Parameter {param.name!r} expects {param.param_type.name.lower()}, got {type(value).__name__}",
            ))

    def _check_range(self, param: ParamSchema, value: Any, report: ConfigValidationReport) -> None:
        """Check numeric parameter is within range."""
        if not isinstance(value, (int, float)):
            return
        if param.min_value is not None and value < param.min_value:
            report.issues.append(ConfigIssue(
                param=param.name, severity="error",
                message=f"Parameter {param.name!r} value {value} is below minimum {param.min_value}",
            ))
        if param.max_value is not None and value > param.max_value:
            report.issues.append(ConfigIssue(
                param=param.name, severity="error",
                message=f"Parameter {param.name!r} value {value} is above maximum {param.max_value}",
            ))

    def _check_allowed(self, param: ParamSchema, value: Any, report: ConfigValidationReport) -> None:
        """Check value is in allowed set."""
        if param.allowed_values is None:
            return
        if value not in param.allowed_values:
            report.issues.append(ConfigIssue(
                param=param.name, severity="error",
                message=f"Parameter {param.name!r} value {value!r} not in allowed: {param.allowed_values}",
            ))

    def get_defaults(self) -> dict[str, Any]:
        """Get a configuration dictionary with all default values."""
        return {p.name: p.default for p in self._schema if p.default is not None}

    def merge_with_defaults(self, config: dict[str, Any]) -> dict[str, Any]:
        """Merge config with defaults (config values take precedence)."""
        merged = self.get_defaults()
        merged.update(config)
        return merged


def validate_config(config: dict[str, Any]) -> ConfigValidationReport:
    """Quick-start: validate an optimizer configuration."""
    return ConfigValidator().validate(config)


def validate_optimizer_defaults() -> ConfigValidationReport:
    """Validate that the default optimizer settings are valid."""
    validator = ConfigValidator()
    return validator.validate(validator.get_defaults())
