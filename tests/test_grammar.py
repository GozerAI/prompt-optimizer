"""Tests for the agent communication grammar: lexer, parser, renderer, compiler."""


from prompt_optimizer.grammar import (
    Compiler,
    ConditionalNode,
    DirectiveNode,
    Lexer,
    Parser,
    PipelineNode,
    Renderer,
    TokenType,
)


# ============================================================
# Lexer Tests
# ============================================================

class TestLexer:
    def setup_method(self):
        self.lexer = Lexer()

    def test_simple_directive(self):
        tokens = self.lexer.tokenize("@CFO ANALYZE revenue")
        types = [t.type for t in tokens]
        assert TokenType.AT in types
        assert TokenType.AGENT_CODE in types
        assert TokenType.ACTION in types

    def test_at_agent(self):
        tokens = self.lexer.tokenize("@CTO")
        assert tokens[0].type == TokenType.AT
        assert tokens[1].type == TokenType.AGENT_CODE
        assert tokens[1].value == "CTO"

    def test_action_recognized(self):
        tokens = self.lexer.tokenize("ANALYZE")
        assert tokens[0].type == TokenType.ACTION

    def test_unknown_word_is_identifier(self):
        tokens = self.lexer.tokenize("foobar")
        assert tokens[0].type == TokenType.IDENTIFIER

    def test_params_braces(self):
        tokens = self.lexer.tokenize("{period=Q1-2026}")
        types = [t.type for t in tokens]
        assert TokenType.LBRACE in types
        assert TokenType.EQUALS in types
        assert TokenType.RBRACE in types

    def test_constraints_brackets(self):
        tokens = self.lexer.tokenize("[within 2h, no deps]")
        types = [t.type for t in tokens]
        assert TokenType.LBRACKET in types
        assert TokenType.COMMA in types
        assert TokenType.RBRACKET in types

    def test_arrow(self):
        tokens = self.lexer.tokenize("-> summary")
        assert tokens[0].type == TokenType.ARROW

    def test_priority_bang(self):
        tokens = self.lexer.tokenize("!urgent")
        assert tokens[0].type == TokenType.PRIORITY

    def test_modifier_tilde(self):
        tokens = self.lexer.tokenize("~thorough")
        assert tokens[0].type == TokenType.MODIFIER

    def test_pipe(self):
        tokens = self.lexer.tokenize("A | B")
        types = [t.type for t in tokens]
        assert TokenType.PIPE in types

    def test_keywords_if_then_else(self):
        tokens = self.lexer.tokenize("IF risk:high THEN ANALYZE ELSE DECIDE")
        types = [t.type for t in tokens]
        assert TokenType.IF in types
        assert TokenType.THEN in types
        assert TokenType.ELSE in types

    def test_comparators(self):
        for op in [">=", "<=", "==", "!=", ">", "<"]:
            tokens = self.lexer.tokenize(f"cost {op} 1000")
            comparator_types = {TokenType.GT, TokenType.LT, TokenType.GTE, TokenType.LTE, TokenType.EQ, TokenType.NEQ}
            assert any(t.type in comparator_types for t in tokens)

    def test_number(self):
        tokens = self.lexer.tokenize("42")
        assert tokens[0].type == TokenType.NUMBER

    def test_dollar_amount(self):
        tokens = self.lexer.tokenize("$2.3M")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "$2.3M"

    def test_quoted_string(self):
        tokens = self.lexer.tokenize('"hello world"')
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello world"

    def test_colon(self):
        tokens = self.lexer.tokenize("risk:high")
        assert tokens[1].type == TokenType.COLON

    def test_full_directive(self):
        tokens = self.lexer.tokenize("@CFO ANALYZE revenue {period=Q1-2026} -> summary !urgent ~thorough")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.AT in types
        assert TokenType.AGENT_CODE in types
        assert TokenType.ACTION in types
        assert TokenType.ARROW in types
        assert TokenType.PRIORITY in types
        assert TokenType.MODIFIER in types

    def test_empty_input(self):
        tokens = self.lexer.tokenize("")
        assert tokens[-1].type == TokenType.EOF

    def test_custom_agent_codes(self):
        lexer = Lexer(agent_codes={"ALPHA", "BETA"})
        tokens = lexer.tokenize("@ALPHA")
        assert tokens[1].type == TokenType.AGENT_CODE


# ============================================================
# Parser Tests
# ============================================================

class TestParser:
    def _parse(self, text: str):
        tokens = Lexer().tokenize(text)
        return Parser(tokens).parse()

    def test_simple_directive(self):
        node = self._parse("@CFO ANALYZE revenue")
        assert isinstance(node, DirectiveNode)
        assert node.action == "ANALYZE"
        assert node.recipient.agent_code == "CFO"
        assert "revenue" in node.target

    def test_directive_with_params(self):
        node = self._parse("@CTO ASSESS risk {scope=infrastructure}")
        assert isinstance(node, DirectiveNode)
        assert node.params is not None
        assert node.params.params[0].key == "scope"
        assert node.params.params[0].value == "infrastructure"

    def test_directive_with_constraints(self):
        node = self._parse("@CTO ASSESS risk [within 2h, no external deps]")
        assert isinstance(node, DirectiveNode)
        assert node.constraints is not None
        assert len(node.constraints.constraints) == 2

    def test_directive_with_output(self):
        node = self._parse("@CFO ANALYZE revenue -> summary")
        assert isinstance(node, DirectiveNode)
        assert node.output is not None
        assert node.output.format == "summary"

    def test_directive_with_priority(self):
        node = self._parse("@CFO ANALYZE revenue !urgent")
        assert isinstance(node, DirectiveNode)
        assert node.priority is not None
        assert node.priority.level == "urgent"

    def test_directive_with_modifiers(self):
        node = self._parse("@CFO ANALYZE revenue ~thorough ~discretion")
        assert isinstance(node, DirectiveNode)
        assert len(node.modifiers) == 2
        assert node.modifiers[0].name == "thorough"

    def test_full_directive(self):
        node = self._parse("@CFO ANALYZE revenue {period=Q1-2026} [within 2h] -> summary !urgent ~thorough")
        assert isinstance(node, DirectiveNode)
        assert node.recipient.agent_code == "CFO"
        assert node.action == "ANALYZE"
        assert node.params is not None
        assert node.constraints is not None
        assert node.output is not None
        assert node.priority is not None
        assert len(node.modifiers) == 1

    def test_pipeline_two_directives(self):
        node = self._parse("@CTO ASSESS risk | @CFO COST estimate")
        assert isinstance(node, PipelineNode)
        assert len(node.directives) == 2

    def test_pipeline_three_directives(self):
        node = self._parse("@CTO ASSESS risk | @CFO COST estimate | @CEO DECIDE go_nogo")
        assert isinstance(node, PipelineNode)
        assert len(node.directives) == 3
        assert node.directives[2].action == "DECIDE"

    def test_conditional_shorthand(self):
        node = self._parse("IF risk:high THEN @CFO ANALYZE costs")
        assert isinstance(node, ConditionalNode)
        assert node.condition.comparator == ":"
        assert node.condition.left.field == "risk"
        assert node.condition.right.field == "high"
        assert isinstance(node.then_branch, DirectiveNode)

    def test_conditional_with_else(self):
        node = self._parse("IF risk:high THEN @CFO ANALYZE costs ELSE @CEO APPROVE deployment")
        assert isinstance(node, ConditionalNode)
        assert node.else_branch is not None
        assert isinstance(node.else_branch, DirectiveNode)

    def test_no_recipient(self):
        node = self._parse("ANALYZE revenue")
        assert isinstance(node, DirectiveNode)
        assert node.recipient is None
        assert node.action == "ANALYZE"

    def test_bare_keyword_param(self):
        node = self._parse("@CFO COST estimate {inherit}")
        assert isinstance(node, DirectiveNode)
        assert node.params.params[0].key == "inherit"
        assert node.params.params[0].value is None


# ============================================================
# Renderer Tests
# ============================================================

class TestRenderer:
    def setup_method(self):
        self.renderer = Renderer()
        self.lexer = Lexer()

    def _roundtrip(self, text: str) -> str:
        tokens = self.lexer.tokenize(text)
        ast = Parser(tokens).parse()
        return self.renderer.render(ast)

    def test_simple_directive_roundtrip(self):
        result = self._roundtrip("@CFO ANALYZE revenue")
        assert "@CFO" in result
        assert "ANALYZE" in result
        assert "revenue" in result

    def test_params_roundtrip(self):
        result = self._roundtrip("@CTO ASSESS risk {scope=infrastructure}")
        assert "scope=infrastructure" in result

    def test_pipeline_roundtrip(self):
        result = self._roundtrip("@CTO ASSESS risk | @CFO COST estimate")
        assert " | " in result

    def test_conditional_roundtrip(self):
        result = self._roundtrip("IF risk:high THEN @CFO ANALYZE costs")
        assert "IF" in result
        assert "THEN" in result
        assert "risk:high" in result

    def test_human_readable_directive(self):
        tokens = self.lexer.tokenize("@CFO ANALYZE revenue")
        ast = Parser(tokens).parse()
        human = self.renderer.render_human(ast)
        assert "CFO" in human
        assert "analyze" in human

    def test_human_readable_pipeline(self):
        tokens = self.lexer.tokenize("@CTO ASSESS risk | @CFO COST estimate | @CEO DECIDE go")
        ast = Parser(tokens).parse()
        human = self.renderer.render_human(ast)
        assert "First" in human
        assert "Then" in human
        assert "Finally" in human


# ============================================================
# Compiler Tests
# ============================================================

class TestCompiler:
    def setup_method(self):
        self.compiler = Compiler()

    def test_simple_nl_to_directive(self):
        node = self.compiler.compile("Analyze the revenue data for Q1 2026.")
        assert isinstance(node, DirectiveNode)
        assert node.action == "ANALYZE"

    def test_extracts_recipient(self):
        node = self.compiler.compile("CFO should analyze the revenue data.")
        assert isinstance(node, DirectiveNode)
        assert node.recipient is not None
        assert node.recipient.agent_code == "CFO"

    def test_extracts_priority(self):
        node = self.compiler.compile("ASAP, analyze the revenue data.")
        assert isinstance(node, DirectiveNode)
        assert node.priority is not None
        assert node.priority.level == "urgent"

    def test_extracts_modifiers(self):
        node = self.compiler.compile("Thoroughly analyze the revenue data.")
        assert isinstance(node, DirectiveNode)
        assert any(m.name == "thorough" for m in node.modifiers)

    def test_extracts_constraints(self):
        node = self.compiler.compile("Analyze revenue. Must complete within 2 hours.")
        assert isinstance(node, DirectiveNode)
        if node.constraints:
            assert len(node.constraints.constraints) >= 1

    def test_multi_step_becomes_pipeline(self):
        text = (
            "First, have the CTO assess the technical risk. "
            "Then, the CFO should estimate costs. "
            "Finally, the CEO needs to decide."
        )
        node = self.compiler.compile(text)
        assert isinstance(node, PipelineNode)
        assert len(node.directives) >= 2

    def test_conditional_nl(self):
        node = self.compiler.compile(
            "If risk is high then analyze mitigation costs"
        )
        assert isinstance(node, ConditionalNode)
        assert node.condition.left.field == "risk"

    def test_conditional_with_else_nl(self):
        node = self.compiler.compile(
            "If risk:high then analyze costs else approve deployment"
        )
        assert isinstance(node, ConditionalNode)
        assert node.else_branch is not None

    def test_no_action_returns_none(self):
        node = self.compiler.compile("Hello world, nice day.")
        assert node is None

    def test_empty_returns_none(self):
        node = self.compiler.compile("")
        assert node is None

    def test_extracts_output_format(self):
        node = self.compiler.compile("Analyze revenue and provide a summary.")
        assert isinstance(node, DirectiveNode)
        if node.output:
            assert node.output.format == "summary"

    def test_extracts_params_period(self):
        node = self.compiler.compile("Analyze revenue for Q1 2026.")
        assert isinstance(node, DirectiveNode)
        if node.params:
            period_params = [p for p in node.params.params if p.key == "period"]
            assert len(period_params) > 0

    def test_compiler_to_renderer_roundtrip(self):
        """Compile NL → AST → compact text → verify readable."""
        node = self.compiler.compile("CFO should analyze the Q1 2026 revenue data urgently.")
        assert node is not None
        renderer = Renderer()
        compact = renderer.render(node)
        assert "ANALYZE" in compact
        assert "CFO" in compact

    def test_pipeline_to_renderer(self):
        text = (
            "First, have the CTO assess risk. "
            "Then, the CFO should estimate costs. "
            "Finally, the CEO needs to decide."
        )
        node = self.compiler.compile(text)
        assert node is not None
        renderer = Renderer()
        compact = renderer.render(node)
        assert "|" in compact
