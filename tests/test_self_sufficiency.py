"""Tests for the self_sufficiency package."""

import pytest


class TestTokenizerBackend:
    def test_backend_enum_values(self):
        from prompt_optimizer.self_sufficiency.offline_mode import TokenizerBackend
        assert TokenizerBackend.TIKTOKEN.value != TokenizerBackend.MEMORY.value

    def test_backend_has_two_members(self):
        from prompt_optimizer.self_sufficiency.offline_mode import TokenizerBackend
        assert len(TokenizerBackend) == 2


class TestOfflineTokenizer:
    def test_memory_backend_creation(self):
        from prompt_optimizer.self_sufficiency.offline_mode import OfflineTokenizer, TokenizerBackend
        tok = OfflineTokenizer(force_backend=TokenizerBackend.MEMORY)
        assert tok.backend == TokenizerBackend.MEMORY
        assert tok.is_offline is True
        assert tok.is_exact is False

    def test_count_tokens_empty(self):
        from prompt_optimizer.self_sufficiency.offline_mode import OfflineTokenizer, TokenizerBackend
        tok = OfflineTokenizer(force_backend=TokenizerBackend.MEMORY)
        assert tok.count_tokens("") == 0

    def test_count_tokens_nonempty(self):
        from prompt_optimizer.self_sufficiency.offline_mode import OfflineTokenizer, TokenizerBackend
        tok = OfflineTokenizer(force_backend=TokenizerBackend.MEMORY)
        assert tok.count_tokens("hello world") > 0

    def test_count_tokens_batch(self):
        from prompt_optimizer.self_sufficiency.offline_mode import OfflineTokenizer, TokenizerBackend
        tok = OfflineTokenizer(force_backend=TokenizerBackend.MEMORY)
        results = tok.count_tokens_batch(["hello", "world", ""])
        assert len(results) == 3
        assert results[0] > 0
        assert results[2] == 0

    def test_estimate_reduction(self):
        from prompt_optimizer.self_sufficiency.offline_mode import OfflineTokenizer, TokenizerBackend
        tok = OfflineTokenizer(force_backend=TokenizerBackend.MEMORY)
        reduction = tok.estimate_reduction("this is a long sentence with many words", "short")
        assert 0.0 < reduction < 1.0

    def test_estimate_reduction_empty(self):
        from prompt_optimizer.self_sufficiency.offline_mode import OfflineTokenizer, TokenizerBackend
        tok = OfflineTokenizer(force_backend=TokenizerBackend.MEMORY)
        assert tok.estimate_reduction("", "anything") == 0.0

    def test_status(self):
        from prompt_optimizer.self_sufficiency.offline_mode import OfflineTokenizer, TokenizerBackend
        tok = OfflineTokenizer(force_backend=TokenizerBackend.MEMORY)
        status = tok.status()
        assert status.offline is True
        assert status.active_backend == TokenizerBackend.MEMORY
        assert len(status.backends) == 2

    def test_unavailable_tiktoken_raises(self):
        from prompt_optimizer.self_sufficiency.offline_mode import OfflineTokenizer, TokenizerBackend
        try:
            import tiktoken
            pytest.skip("tiktoken is installed")
        except ImportError:
            with pytest.raises(RuntimeError):
                OfflineTokenizer(force_backend=TokenizerBackend.TIKTOKEN)


class TestOfflineModuleFunctions:
    def test_singleton_and_reset(self):
        from prompt_optimizer.self_sufficiency.offline_mode import get_offline_tokenizer, reset_offline_tokenizer
        tok1 = get_offline_tokenizer()
        tok2 = get_offline_tokenizer()
        assert tok1 is tok2
        reset_offline_tokenizer()
        tok3 = get_offline_tokenizer()
        assert tok3 is not tok1
        reset_offline_tokenizer()

    def test_count_tokens_offline(self):
        from prompt_optimizer.self_sufficiency.offline_mode import count_tokens_offline, reset_offline_tokenizer
        reset_offline_tokenizer()
        assert count_tokens_offline("hello world test") > 0
        reset_offline_tokenizer()

    def test_is_offline(self):
        from prompt_optimizer.self_sufficiency.offline_mode import is_offline, reset_offline_tokenizer
        reset_offline_tokenizer()
        assert isinstance(is_offline(), bool)
        reset_offline_tokenizer()


class TestGrammarIssue:
    def test_creation(self):
        from prompt_optimizer.self_sufficiency.grammar_validator import GrammarIssue
        issue = GrammarIssue(category="conflict", severity="error", message="test")
        assert issue.category == "conflict"
        assert issue.details == {}

    def test_with_details(self):
        from prompt_optimizer.self_sufficiency.grammar_validator import GrammarIssue
        issue = GrammarIssue(category="missing", severity="warning", message="test", details={"k": "v"})
        assert issue.details["k"] == "v"


class TestGrammarValidationReport:
    def test_empty_is_valid(self):
        from prompt_optimizer.self_sufficiency.grammar_validator import GrammarValidationReport
        r = GrammarValidationReport()
        assert r.valid is True
        assert r.error_count == 0
        assert r.warning_count == 0

    def test_with_error_is_invalid(self):
        from prompt_optimizer.self_sufficiency.grammar_validator import GrammarValidationReport, GrammarIssue
        r = GrammarValidationReport(issues=[GrammarIssue(category="t", severity="error", message="f")])
        assert r.valid is False
        assert r.error_count == 1

    def test_with_warning_is_valid(self):
        from prompt_optimizer.self_sufficiency.grammar_validator import GrammarValidationReport, GrammarIssue
        r = GrammarValidationReport(issues=[GrammarIssue(category="t", severity="warning", message="w")])
        assert r.valid is True
        assert r.warning_count == 1


class TestGrammarConsistencyValidator:
    def test_validate_returns_report(self):
        from prompt_optimizer.self_sufficiency.grammar_validator import GrammarConsistencyValidator
        report = GrammarConsistencyValidator().validate()
        assert report.checks_run >= 8

    def test_current_grammar_is_valid(self):
        from prompt_optimizer.self_sufficiency.grammar_validator import validate_grammar
        report = validate_grammar()
        assert report.valid is True

    def test_is_grammar_valid_function(self):
        from prompt_optimizer.self_sufficiency.grammar_validator import is_grammar_valid
        assert is_grammar_valid() is True

    def test_get_import_validation_caches(self):
        from prompt_optimizer.self_sufficiency.grammar_validator import get_import_validation
        r1 = get_import_validation()
        r2 = get_import_validation()
        assert r1 is r2

    def test_no_keyword_action_overlap(self):
        from prompt_optimizer.self_sufficiency.grammar_validator import validate_grammar
        report = validate_grammar()
        overlaps = [i for i in report.issues if i.category == "overlap"]
        assert len(overlaps) == 0

    def test_agent_code_warnings_are_not_errors(self):
        from prompt_optimizer.self_sufficiency.grammar_validator import validate_grammar
        report = validate_grammar()
        # Mixed-case agent codes (CComO, CEngO, etc.) produce warnings, not errors
        case_issues = [i for i in report.issues if "uppercase" in i.message]
        for issue in case_issues:
            assert issue.severity == "warning"
        # No agent code should produce an error
        code_errors = [i for i in report.issues if i.category == "conflict" and i.severity == "error"]
        assert len(code_errors) == 0

    def test_priority_levels_complete(self):
        from prompt_optimizer.self_sufficiency.grammar_validator import validate_grammar
        report = validate_grammar()
        missing = [i for i in report.issues if "priority level" in i.message]
        assert len(missing) == 0


class TestPassMetadata:
    def test_creation(self):
        from prompt_optimizer.self_sufficiency.self_documenting import PassMetadata
        pm = PassMetadata(name="test", level=1, description="Test",
                          expected_reduction=(0.5, 0.7), risk_range=(0.01, 0.05), reversible=True)
        assert pm.name == "test"
        assert pm.transformations == []


class TestPassDocumenter:
    def test_collect_passes(self):
        from prompt_optimizer.self_sufficiency.self_documenting import PassDocumenter
        passes = PassDocumenter().collect_pass_metadata()
        assert len(passes) == 3
        names = {p.name for p in passes}
        assert "structural" in names
        assert "semantic" in names
        assert "contextual" in names

    def test_pass_levels(self):
        from prompt_optimizer.self_sufficiency.self_documenting import PassDocumenter
        passes = PassDocumenter().collect_pass_metadata()
        assert {p.level for p in passes} == {1, 2, 3}

    def test_pass_risk_ranges(self):
        from prompt_optimizer.self_sufficiency.self_documenting import PassDocumenter
        for p in PassDocumenter().collect_pass_metadata():
            assert p.risk_range[0] >= 0.0
            assert p.risk_range[1] <= 1.0

    def test_generate_markdown(self):
        from prompt_optimizer.self_sufficiency.self_documenting import PassDocumenter
        md = PassDocumenter().generate_markdown()
        assert "# Prompt Optimizer" in md
        assert "## Summary" in md
        assert "Structural" in md

    def test_generate_markdown_has_transformations(self):
        from prompt_optimizer.self_sufficiency.self_documenting import PassDocumenter
        md = PassDocumenter().generate_markdown()
        assert "### Transformations" in md
        assert "Filler word removal" in md

    def test_generate_markdown_has_dependencies(self):
        from prompt_optimizer.self_sufficiency.self_documenting import PassDocumenter
        md = PassDocumenter().generate_markdown()
        assert "### Dependencies" in md

    def test_generate_pass_summary_irreversible(self):
        from prompt_optimizer.self_sufficiency.self_documenting import PassDocumenter, PassMetadata
        pm = PassMetadata(name="test", level=1, description="Test",
                          expected_reduction=(0.5, 0.7), risk_range=(0.01, 0.05), reversible=False)
        s = PassDocumenter().generate_pass_summary(pm)
        assert "irreversible" in s

    def test_generate_docs_function(self):
        from prompt_optimizer.self_sufficiency.self_documenting import generate_docs
        assert len(generate_docs()) > 100

    def test_list_passes_function(self):
        from prompt_optimizer.self_sufficiency.self_documenting import list_passes
        assert len(list_passes()) == 3

    def test_contextual_not_reversible(self):
        from prompt_optimizer.self_sufficiency.self_documenting import list_passes
        ctx = [p for p in list_passes() if p.name == "contextual"][0]
        assert ctx.reversible is False

    def test_structural_is_reversible(self):
        from prompt_optimizer.self_sufficiency.self_documenting import list_passes
        struct = [p for p in list_passes() if p.name == "structural"][0]
        assert struct.reversible is True


class TestParamType:
    def test_enum_values(self):
        from prompt_optimizer.self_sufficiency.config_validator import ParamType
        assert len(ParamType) == 5


class TestConfigValidationReport:
    def test_empty_is_valid(self):
        from prompt_optimizer.self_sufficiency.config_validator import ConfigValidationReport
        r = ConfigValidationReport()
        assert r.valid is True
        assert r.error_count == 0


class TestConfigValidator:
    def test_valid_config(self):
        from prompt_optimizer.self_sufficiency.config_validator import validate_config
        assert validate_config({"min_fidelity": 0.5, "max_risk": 0.25}).valid is True

    def test_defaults_are_valid(self):
        from prompt_optimizer.self_sufficiency.config_validator import validate_optimizer_defaults
        assert validate_optimizer_defaults().valid is True

    def test_out_of_range_value(self):
        from prompt_optimizer.self_sufficiency.config_validator import validate_config
        r = validate_config({"min_fidelity": 2.0})
        assert r.valid is False
        assert any("above maximum" in i.message for i in r.issues)

    def test_negative_value(self):
        from prompt_optimizer.self_sufficiency.config_validator import validate_config
        r = validate_config({"min_fidelity": -0.5})
        assert r.valid is False

    def test_wrong_type(self):
        from prompt_optimizer.self_sufficiency.config_validator import validate_config
        r = validate_config({"max_risk": "not_a_float"})
        assert r.valid is False
        assert any("expects float" in i.message for i in r.issues)

    def test_unknown_param_warning(self):
        from prompt_optimizer.self_sufficiency.config_validator import validate_config
        r = validate_config({"unknown_xyz": True})
        assert r.warning_count > 0

    def test_max_layer_range(self):
        from prompt_optimizer.self_sufficiency.config_validator import validate_config
        assert validate_config({"max_layer": 5}).valid is False

    def test_cross_field_fidelity_risk(self):
        from prompt_optimizer.self_sufficiency.config_validator import validate_config
        r = validate_config({"min_fidelity": 0.1, "max_risk": 0.9})
        assert any("fidelity" in i.message.lower() for i in r.issues if i.severity == "warning")

    def test_cross_field_layer_risk(self):
        from prompt_optimizer.self_sufficiency.config_validator import validate_config
        r = validate_config({"max_layer": 3, "max_risk": 0.05})
        assert any("Layer 3" in i.message for i in r.issues if i.severity == "warning")

    def test_cross_field_reduction_layer(self):
        from prompt_optimizer.self_sufficiency.config_validator import validate_config
        r = validate_config({"target_reduction": 0.95, "max_layer": 1})
        assert any("unlikely" in i.message.lower() for i in r.issues if i.severity == "warning")

    def test_get_defaults(self):
        from prompt_optimizer.self_sufficiency.config_validator import ConfigValidator
        defaults = ConfigValidator().get_defaults()
        assert defaults["min_fidelity"] == 0.50
        assert defaults["max_risk"] == 0.25

    def test_merge_with_defaults(self):
        from prompt_optimizer.self_sufficiency.config_validator import ConfigValidator
        merged = ConfigValidator().merge_with_defaults({"min_fidelity": 0.9})
        assert merged["min_fidelity"] == 0.9
        assert merged["max_risk"] == 0.25

    def test_schema_property(self):
        from prompt_optimizer.self_sufficiency.config_validator import ConfigValidator
        schema = ConfigValidator().schema
        assert len(schema) >= 7

    def test_empty_config_valid(self):
        from prompt_optimizer.self_sufficiency.config_validator import validate_config
        assert validate_config({}).valid is True

    def test_list_type_invalid(self):
        from prompt_optimizer.self_sufficiency.config_validator import validate_config
        assert validate_config({"agent_codes": "not_a_list"}).valid is False

    def test_list_type_valid(self):
        from prompt_optimizer.self_sufficiency.config_validator import validate_config
        assert validate_config({"agent_codes": ["CEO", "CFO"]}).valid is True


class TestPackageInit:
    def test_all_exports(self):
        import prompt_optimizer.self_sufficiency as ss
        for name in ss.__all__:
            assert hasattr(ss, name), f"Missing export: {name}"

    def test_import_count(self):
        import prompt_optimizer.self_sufficiency as ss
        assert len(ss.__all__) >= 20
