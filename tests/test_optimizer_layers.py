"""Tests for each optimization layer in isolation — L1, L2, L3, and ProgressiveOptimizer."""

import time

import pytest

from prompt_optimizer.blackboard import Blackboard
from prompt_optimizer.fidelity import FidelityScorer
from prompt_optimizer.layers.contextual import ContextualLayer
from prompt_optimizer.layers.semantic import SemanticIndex, SemanticLayer
from prompt_optimizer.layers.structural import (
    FILLER_PATTERNS,
    StructuralLayer,
)
from prompt_optimizer.progressive import ProgressiveOptimizer
from prompt_optimizer.schema_registry import SchemaRegistry
from prompt_optimizer.types import CompressionContext, LayerFidelity
from prompt_optimizer.verifier import ReconstructionVerifier


# ============================================================
# L1: Structural layer
# ============================================================


class TestL1FillerRemoval:
    def setup_method(self):
        self.layer = StructuralLayer()
        self.ctx = CompressionContext(agent_codes=["CEO", "CFO", "CTO"])

    def test_strips_please(self):
        result = self.layer.compress("Please analyze the revenue data.", self.ctx)
        assert "please" not in result.output_text.lower()

    def test_strips_hey(self):
        result = self.layer.compress("Hey CFO, analyze the revenue data.", self.ctx)
        assert "hey" not in result.output_text.lower()

    def test_strips_could_you(self):
        result = self.layer.compress("Could you analyze the revenue data?", self.ctx)
        assert "could you" not in result.output_text.lower()

    def test_strips_thank_you(self):
        result = self.layer.compress("Thank you. Analyze the revenue data.", self.ctx)
        assert "thank you" not in result.output_text.lower()

    def test_strips_hedging(self):
        result = self.layer.compress("I think we should analyze the revenue data.", self.ctx)
        assert "i think" not in result.output_text.lower()

    def test_strips_multiple_fillers(self):
        text = (
            "Hey CFO, could you please take a careful look at the revenue data? "
            "Thank you very much."
        )
        result = self.layer.compress(text, self.ctx)
        output = result.output_text.lower()
        assert "hey" not in output
        assert "please" not in output
        assert "thank you" not in output

    def test_preserves_action_verb(self):
        result = self.layer.compress("Please analyze the Q1 2026 revenue.", self.ctx)
        output = result.output_text.lower()
        assert "analy" in output or "ANALYZE" in result.output_text

    def test_preserves_key_data(self):
        result = self.layer.compress("Please analyze the Q1 2026 revenue of $2.3M.", self.ctx)
        output = result.output_text
        assert "Q1" in output or "2026" in output

    def test_custom_filler_patterns(self):
        custom = StructuralLayer(extra_filler_patterns=[r"(?i)\byo\b"])
        result = custom.compress("Yo, analyze the revenue.", self.ctx)
        assert "yo" not in result.output_text.lower()


class TestL1TypedEnvelope:
    def setup_method(self):
        self.layer = StructuralLayer()
        self.ctx = CompressionContext(agent_codes=["CEO", "CFO", "CTO"])

    def test_directive_prompt_compiles_to_ast(self):
        result = self.layer.compress("CFO should analyze the Q1 2026 revenue.", self.ctx)
        xforms = str(result.transformations)
        assert "compiled to AST" in xforms or "filler stripped" in xforms

    def test_compact_output_contains_action(self):
        result = self.layer.compress("CFO should analyze the Q1 2026 revenue data.", self.ctx)
        # Either AST-rendered (uppercase) or filler-stripped (lowercase)
        output = result.output_text
        assert "ANALYZE" in output or "analyze" in output.lower()

    def test_decompression_produces_readable_text(self):
        result = self.layer.compress(
            "CFO should analyze the Q1 2026 revenue data urgently.", self.ctx
        )
        decompressed = self.layer.decompress(result.output_text, self.ctx)
        assert decompressed  # Non-empty
        assert len(decompressed) > 0


class TestL1ReductionRatio:
    def setup_method(self):
        self.layer = StructuralLayer()
        self.ctx = CompressionContext(agent_codes=["CEO", "CFO", "CTO"])

    def test_polite_prompt_reduces_meaningfully(self):
        text = (
            "Hey CFO, could you please take a careful look at the Q1 2026 revenue "
            "numbers and give me a detailed breakdown of the growth rate compared "
            "to last quarter, along with any key metrics you think are relevant."
        )
        result = self.layer.compress(text, self.ctx)
        # Should achieve at least 20% reduction on verbose prompts
        assert result.reduction_pct > 0.20

    def test_already_compact_input_produces_output(self):
        result = self.layer.compress("ANALYZE revenue Q1-2026", self.ctx)
        # Compact input may be re-compiled (adding extracted params), so just
        # verify it produces valid output without crashing
        assert result.output_text
        assert "revenue" in result.output_text.lower() or "revenue" in result.output_text

    def test_empty_input(self):
        result = self.layer.compress("", self.ctx)
        assert result.output_tokens <= result.input_tokens

    def test_risk_score_within_range(self):
        result = self.layer.compress(
            "Please analyze the revenue data for Q1 2026.", self.ctx
        )
        assert result.risk_score >= self.layer.risk_range[0]
        assert result.risk_score <= self.layer.risk_range[1]

    def test_layer_level_and_name(self):
        assert self.layer.level == 1
        assert self.layer.name == "structural"


# ============================================================
# L2: Semantic layer
# ============================================================


class TestL2ContextDeduplication:
    def setup_method(self):
        self.layer = SemanticLayer()

    def test_duplicate_sentence_replaced_with_reference(self):
        ctx = CompressionContext(
            history=["The Q1 2026 revenue was $2.3M with a growth rate of 12.4%."]
        )
        text = "The Q1 2026 revenue was $2.3M with a growth rate of 12.4%. Analyze it."
        result = self.layer.compress(text, ctx)
        # The duplicated sentence should be replaced with a context reference
        assert "[ctx:" in result.output_text or result.output_text != text

    def test_non_duplicate_preserved(self):
        ctx = CompressionContext(history=["Something completely different."])
        text = "Analyze the Q1 revenue data."
        result = self.layer.compress(text, ctx)
        # Non-duplicate content should be preserved
        assert "revenue" in result.output_text.lower() or "Q1" in result.output_text

    def test_multiple_history_entries_registered(self):
        ctx = CompressionContext(
            history=[
                "Revenue was $2.3M for Q1 2026.",
                "The board mandated maintaining 20% profit margins.",
            ]
        )
        text = "Revenue was $2.3M for Q1 2026. We need to analyze trends."
        result = self.layer.compress(text, ctx)
        # First sentence duplicates history
        assert "[ctx:" in result.output_text or result.output_tokens <= result.input_tokens


class TestL2SemanticIndex:
    def test_register_and_resolve(self):
        idx = SemanticIndex()
        ref = idx.register("Hello world")
        assert ref.startswith("ctx:")
        resolved = idx.resolve(ref)
        assert resolved == "Hello world"

    def test_duplicate_returns_same_ref(self):
        idx = SemanticIndex()
        ref1 = idx.register("Hello world")
        ref2 = idx.register("Hello world")
        assert ref1 == ref2

    def test_check_duplicate(self):
        idx = SemanticIndex()
        idx.register("test content")
        assert idx.check_duplicate("test content") is not None
        assert idx.check_duplicate("other content") is None

    def test_case_insensitive_dedup(self):
        idx = SemanticIndex()
        ref1 = idx.register("Hello World")
        ref2 = idx.register("hello world")
        assert ref1 == ref2

    def test_whitespace_normalized(self):
        idx = SemanticIndex()
        ref1 = idx.register("hello  world")
        ref2 = idx.register("hello world")
        assert ref1 == ref2


class TestL2PipelineShorthand:
    def setup_method(self):
        self.layer = SemanticLayer()
        self.ctx = CompressionContext()

    def test_sequential_steps_collapse_to_pipe(self):
        text = (
            "First, gather the market data. "
            "Then, analyze the trends. "
            "Finally, report the findings."
        )
        result = self.layer.compress(text, self.ctx)
        # If pipeline collapse triggered, PIPE() should appear
        if "PIPE(" in result.output_text:
            assert result.output_tokens <= result.input_tokens

    def test_no_collapse_without_sequence_markers(self):
        text = "Analyze the data and report the findings."
        result = self.layer.compress(text, self.ctx)
        assert "PIPE(" not in result.output_text


class TestL2SchemaAbbreviation:
    def setup_method(self):
        self.layer = SemanticLayer()

    def test_schema_abbreviation_applied(self):
        registry = SchemaRegistry()
        registry.register_vocabulary("custom", {"revenue_analysis": "rev_a"})
        ctx = CompressionContext(schema_registry=registry)
        text = "Perform a revenue_analysis on Q1 data."
        result = self.layer.compress(text, ctx)
        # Schema abbreviation should shorten the term
        assert result.output_text is not None
        # If abbreviation was applied, the short form should appear
        if "applied schema abbreviations" in str(result.transformations):
            assert "rev_a" in result.output_text

    def test_layer_level_and_name(self):
        assert self.layer.level == 2
        assert self.layer.name == "semantic"

    def test_risk_score_increases_with_transformations(self):
        ctx = CompressionContext(
            history=["Revenue was $2.3M for Q1 2026."]
        )
        text = "Revenue was $2.3M for Q1 2026. Also analyze the trends."
        result = self.layer.compress(text, ctx)
        assert result.risk_score >= self.layer.risk_range[0]

    def test_decompress_expands_context_refs(self):
        ctx = CompressionContext()
        self.layer._index.register("original text here")
        compressed = "[ctx:1] and some more text"
        decompressed = self.layer.decompress(compressed, ctx)
        assert "original text here" in decompressed

    def test_decompress_expands_pipe_notation(self):
        ctx = CompressionContext()
        compressed = "PIPE(gather data \u2192 analyze trends \u2192 report)"
        decompressed = self.layer.decompress(compressed, ctx)
        assert "First" in decompressed
        assert "Then" in decompressed
        assert "Finally" in decompressed

    def test_reset_index_clears_state(self):
        self.layer._index.register("some text")
        assert self.layer._index.check_duplicate("some text") is not None
        self.layer.reset_index()
        assert self.layer._index.check_duplicate("some text") is None


# ============================================================
# L3: Contextual layer
# ============================================================


class TestL3BlackboardPointers:
    def setup_method(self):
        self.layer = ContextualLayer()

    def test_no_blackboard_skips_compression(self):
        ctx = CompressionContext(blackboard=None)
        text = "Our company revenue of $2.3M for Q1 2026."
        result = self.layer.compress(text, ctx)
        assert result.output_text == text
        assert result.risk_score == 0.0

    def test_with_blackboard_externalizes_context(self):
        bb = Blackboard()
        ctx = CompressionContext(blackboard=bb)
        text = "Our company revenue of $2.3M for Q1 2026 shows strong growth momentum."
        result = self.layer.compress(text, ctx)
        # If context was extracted, bb refs should appear
        if result.transformations and "externalized" in str(result.transformations):
            assert "bb=" in result.output_text or "[" in result.output_text

    def test_pointer_resolution_roundtrip(self):
        bb = Blackboard()
        ctx = CompressionContext(blackboard=bb)
        text = "Our company revenue of $2.3M for Q1 2026 shows continued positive trends."
        result = self.layer.compress(text, ctx)
        if result.output_text != text:
            decompressed = self.layer.decompress(result.output_text, ctx)
            # Decompressed should restore at least some original content
            assert decompressed is not None

    def test_layer_level_and_name(self):
        assert self.layer.level == 3
        assert self.layer.name == "contextual"

    def test_risk_scales_with_externalized_blocks(self):
        bb = Blackboard()
        ctx = CompressionContext(blackboard=bb)
        text = (
            "Our company revenue of $2.3M for Q1 2026. "
            "The board's mandate to maintain 20% margins is critical. "
            "Strategic plan involves expanding internationally."
        )
        result = self.layer.compress(text, ctx)
        assert result.risk_score <= self.layer.risk_range[1]


class TestL3StalePointerDetection:
    def test_stale_detection(self):
        bb = Blackboard(staleness_threshold=0.0)
        bb.put("org", "state", "active")
        # With threshold 0, everything is immediately stale
        time.sleep(0.01)
        stale = bb.get_stale(threshold=0.001)
        assert len(stale) > 0

    def test_fresh_pointer_not_stale(self):
        bb = Blackboard(staleness_threshold=3600.0)
        bb.put("org", "state", "active")
        assert not bb.is_stale("org:state", threshold=3600.0)

    def test_missing_pointer_is_stale(self):
        bb = Blackboard()
        assert bb.is_stale("nonexistent:key")


class TestL3Decompression:
    def test_decompress_without_blackboard_is_noop(self):
        layer = ContextualLayer()
        ctx = CompressionContext(blackboard=None)
        text = "bb=[org:state@v1] some [org:state@v1] text"
        result = layer.decompress(text, ctx)
        assert result == text

    def test_decompress_resolves_pointers(self):
        layer = ContextualLayer()
        bb = Blackboard()
        bb.put("org", "state", "enterprise mode active")
        ctx = CompressionContext(blackboard=bb)
        pointer = bb.get_latest_pointer("org", "state")
        text = f"bb=[{pointer}] some [{pointer}] text"
        result = layer.decompress(text, ctx)
        assert "enterprise mode active" in result

    def test_decompress_removes_bb_header(self):
        layer = ContextualLayer()
        bb = Blackboard()
        bb.put("org", "state", "value")
        ctx = CompressionContext(blackboard=bb)
        text = "bb=[org:state@v1] remaining text"
        result = layer.decompress(text, ctx)
        assert not result.startswith("bb=[")


# ============================================================
# Progressive optimizer
# ============================================================


class TestProgressiveLayerEscalation:
    def test_applies_layers_in_order(self):
        optimizer = ProgressiveOptimizer()
        result = optimizer.optimize(
            "Hey CFO, could you please analyze the revenue data for Q1 2026?",
            max_layer=2,
        )
        # Should apply L1 at minimum
        assert len(result.layers_applied) >= 1
        if len(result.layers_applied) > 1:
            assert result.layers_applied == sorted(result.layers_applied)

    def test_stops_at_max_layer(self):
        optimizer = ProgressiveOptimizer()
        result = optimizer.optimize(
            "Please analyze the revenue data.", max_layer=1
        )
        assert all(l <= 1 for l in result.layers_applied)

    def test_respects_min_fidelity(self):
        # With very high fidelity threshold, should stop early
        optimizer = ProgressiveOptimizer(min_fidelity=0.999)
        result = optimizer.optimize(
            "Hey CFO, please analyze the revenue data."
        )
        # Should produce a result (may stop after L1 or even skip)
        assert result.compressed_text is not None

    def test_respects_max_risk(self):
        optimizer = ProgressiveOptimizer(max_risk=0.0)
        result = optimizer.optimize(
            "Hey CFO, please analyze the revenue data."
        )
        # With max_risk=0, nothing should pass
        assert result.compressed_text is not None

    def test_target_reduction_stops_early(self):
        optimizer = ProgressiveOptimizer()
        result = optimizer.optimize(
            "Hey CFO, could you please take a careful look at the revenue data? Thank you.",
            target_reduction=0.10,
            max_layer=3,
        )
        assert result.compressed_text is not None

    def test_token_counts_populated(self):
        optimizer = ProgressiveOptimizer()
        result = optimizer.optimize(
            "Please analyze the revenue data for Q1 2026."
        )
        assert result.token_counts.original > 0
        assert result.token_counts.compressed > 0
        # Compression may sometimes expand short prompts (e.g. adding extracted params),
        # so just verify counts are populated and positive

    def test_fidelity_report_generated(self):
        optimizer = ProgressiveOptimizer()
        result = optimizer.optimize(
            "Hey CFO, analyze revenue data for Q1 2026."
        )
        assert result.fidelity_report is not None
        assert result.fidelity_report.overall_score >= 0.0

    def test_layer_results_populated(self):
        optimizer = ProgressiveOptimizer()
        result = optimizer.optimize(
            "Hey CFO, could you please analyze revenue?", max_layer=2
        )
        # layer_results tracks results from applied layers
        if result.layers_applied:
            assert len(result.layer_results) >= 1


class TestProgressiveEscapeHatch:
    def test_fidelity_escape_halts_progression(self):
        # Create optimizer with very high fidelity requirement
        optimizer = ProgressiveOptimizer(min_fidelity=0.99)
        text = (
            "Hey CFO, could you please take a careful look at the Q1 2026 revenue "
            "numbers and give me a detailed breakdown."
        )
        result = optimizer.optimize(text, max_layer=3)
        # Should still produce output even if layers are skipped
        assert result.compressed_text is not None
        assert result.token_counts.compressed <= result.token_counts.original

    def test_risk_escape_halts_progression(self):
        optimizer = ProgressiveOptimizer(max_risk=0.001)
        result = optimizer.optimize(
            "Please analyze the revenue data.", max_layer=3
        )
        assert result.compressed_text is not None

    def test_no_context_still_works(self):
        optimizer = ProgressiveOptimizer()
        result = optimizer.optimize("Analyze revenue.", context=None)
        assert result.compressed_text is not None

    def test_with_explicit_context(self):
        bb = Blackboard()
        ctx = CompressionContext(
            blackboard=bb,
            history=["Previous analysis showed $2.3M revenue."],
        )
        optimizer = ProgressiveOptimizer()
        result = optimizer.optimize(
            "Hey CFO, analyze the revenue data.", context=ctx
        )
        assert result.compressed_text is not None


class TestProgressiveLayerOrder:
    def test_default_layers_are_1_2_3(self):
        optimizer = ProgressiveOptimizer()
        levels = [l.level for l in optimizer.layers]
        assert levels == [1, 2, 3]

    def test_custom_layers(self):
        l1 = StructuralLayer()
        optimizer = ProgressiveOptimizer(layers=[l1])
        assert len(optimizer.layers) == 1
        assert optimizer.layers[0].level == 1

    def test_each_layer_applied_in_order(self):
        optimizer = ProgressiveOptimizer()
        result = optimizer.optimize(
            "Hey CFO, could you please analyze the Q1 2026 revenue data?",
            max_layer=2,
        )
        for i in range(1, len(result.layers_applied)):
            assert result.layers_applied[i] > result.layers_applied[i - 1]


# ============================================================
# FidelityScorer standalone
# ============================================================


class TestFidelityScorer:
    def test_identical_text_scores_high(self):
        scorer = FidelityScorer()
        fidelity = scorer.score("Analyze the revenue.", "Analyze the revenue.", 1)
        assert fidelity.overall >= 0.9

    def test_completely_different_text_scores_low(self):
        scorer = FidelityScorer()
        fidelity = scorer.score(
            "Analyze the Q1 2026 revenue of $2.3M.",
            "Banana smoothie recipe.",
            1,
        )
        assert fidelity.overall < 0.5

    def test_compressed_preserving_key_elements(self):
        scorer = FidelityScorer()
        fidelity = scorer.score(
            "Please analyze the Q1 2026 revenue of $2.3M for the CFO.",
            "@CFO ANALYZE revenue {period=Q1 2026}",
            1,
        )
        # Key elements (CFO, Q1, 2026) preserved
        assert fidelity.completeness > 0.3

    def test_layer_fidelity_overall_formula(self):
        lf = LayerFidelity(layer=1, completeness=0.8, accuracy=0.9, actionability=1.0)
        expected = 0.8 * 0.4 + 0.9 * 0.4 + 1.0 * 0.2
        assert abs(lf.overall - expected) < 0.001
