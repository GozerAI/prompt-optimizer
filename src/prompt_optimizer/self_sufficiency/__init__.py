"""Self-sufficiency module -- offline operation, self-validation, and self-documentation."""

from prompt_optimizer.self_sufficiency.config_validator import (
    ConfigIssue,
    ConfigValidationReport,
    ConfigValidator,
    ParamSchema,
    ParamType,
    validate_config,
    validate_optimizer_defaults,
)
from prompt_optimizer.self_sufficiency.grammar_validator import (
    GrammarConsistencyValidator,
    GrammarIssue,
    GrammarValidationReport,
    get_import_validation,
    is_grammar_valid,
    validate_grammar,
)
from prompt_optimizer.self_sufficiency.offline_mode import (
    OfflineModeStatus,
    OfflineTokenizer,
    TokenizerBackend,
    count_tokens_offline,
    get_offline_tokenizer,
    is_offline,
    reset_offline_tokenizer,
)
from prompt_optimizer.self_sufficiency.self_documenting import (
    PassDocumenter,
    PassMetadata,
    generate_docs,
    list_passes,
)

__all__ = [
    "OfflineModeStatus", "OfflineTokenizer", "TokenizerBackend",
    "count_tokens_offline", "get_offline_tokenizer", "is_offline", "reset_offline_tokenizer",
    "GrammarConsistencyValidator", "GrammarIssue", "GrammarValidationReport",
    "get_import_validation", "is_grammar_valid", "validate_grammar",
    "PassDocumenter", "PassMetadata", "generate_docs", "list_passes",
    "ConfigIssue", "ConfigValidationReport", "ConfigValidator",
    "ParamSchema", "ParamType", "validate_config", "validate_optimizer_defaults",
]
