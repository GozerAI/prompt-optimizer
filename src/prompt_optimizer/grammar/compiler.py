"""Compiler — converts natural language to AST nodes.

This is the bridge between free-form human text and the structured grammar.
Uses pattern matching to extract structure, then builds proper AST nodes.
"""

from __future__ import annotations

import re

from prompt_optimizer.grammar.ast_nodes import (
    ASTNode,
    ConditionNode,
    ConditionalNode,
    ConstraintNode,
    ConstraintsNode,
    DirectiveNode,
    ExpressionNode,
    ModifierNode,
    OutputNode,
    ParamNode,
    ParamsNode,
    PipelineNode,
    PriorityNode,
    RecipientNode,
)

# Action verb patterns → canonical action names
ACTION_MAP: list[tuple[str, str]] = [
    (r"(?i)\b(analyz[es]*|assess(?:es)?|evaluat[es]*|examin[es]*|review(?:s)?)\b", "ANALYZE"),
    (r"(?i)\b(look\s+at|breakdown|break\s+down|inspect(?:s)?)\b", "ANALYZE"),
    (r"(?i)\b(generat[es]*|creat[es]*|produc[es]*|build(?:s)?|draft(?:s)?|write(?:s)?)\b", "GENERATE"),
    (r"(?i)\b(decid[es]*|determin[es]*|choose(?:s)?|select(?:s)?)\b", "DECIDE"),
    (r"(?i)\b(summariz[es]*|condense(?:s)?|distill(?:s)?|recap(?:s)?)\b", "SUMMARIZE"),
    (r"(?i)\b(delegat[es]*|assign(?:s)?|hand\s+off|pass\s+to|forward\s+to)\b", "DELEGATE"),
    (r"(?i)\b(recommend(?:s)?|suggest(?:s)?|propos[es]*|advis[es]*)\b", "RECOMMEND"),
    (r"(?i)\b(forecast(?:s)?|predict(?:s)?|project(?:s)?|estimat[es]*)\b", "FORECAST"),
    (r"(?i)\b(report(?:s)?|update(?:s)?|inform(?:s)?|notify|brief(?:s)?)\b", "REPORT"),
    (r"(?i)\b(plan(?:s)?|schedul[es]*|organiz[es]*|coordinat[es]*)\b", "PLAN"),
    (r"(?i)\b(monitor(?:s)?|track(?:s)?|watch(?:es)?|observe(?:s)?|check(?:s)?)\b", "MONITOR"),
    (r"(?i)\b(optimiz[es]*|improv[es]*|enhanc[es]*|refin[es]*)\b", "OPTIMIZE"),
    (r"(?i)\b(approv[es]*|authoriz[es]*|sign\s+off|green[- ]light)\b", "APPROVE"),
    (r"(?i)\b(cost(?:s)?|pric[es]*|budget(?:s)?)\b", "COST"),
]

# Agent code patterns
AGENT_PATTERNS = [
    r"(?i)\b(CEO|COO|CTO|CFO|CIO|CMO|CHRO|CLO|CSO|CDO|CPO|CRO)\b",
    r"(?i)\b(CCO|CSecO|CComO|CEngO|CRiO|CRevO|CCoMO)\b",
]

# Priority patterns
PRIORITY_MAP: list[tuple[str, str]] = [
    (r"(?i)\b(ASAP|immediately|urgent(?:ly)?|right\s+away|top\s+priority)\b", "urgent"),
    (r"(?i)\b(high\s+priority|important|critical)\b", "high"),
    (r"(?i)\b(when\s+you\s+(?:get\s+a\s+chance|can)|low\s+priority|no\s+rush)\b", "low"),
]

# Modifier patterns
MODIFIER_MAP: list[tuple[str, str]] = [
    (r"(?i)\b(use\s+your\s+(?:judgment|discretion|expertise))\b", "discretion"),
    (r"(?i)\b(you\s+think\s+are\s+relevant|as\s+you\s+see\s+fit)\b", "discretion"),
    (r"(?i)\b(thorough(?:ly)?|in[- ]depth|comprehensive(?:ly)?|detail(?:ed)?)\b", "thorough"),
    (r"(?i)\b(brief(?:ly)?|quick(?:ly)?|high[- ]level|overview)\b", "brief"),
]

# Output format patterns
OUTPUT_MAP: list[tuple[str, str]] = [
    (r"(?i)\b(breakdown|itemized|line[- ]by[- ]line)\b", "breakdown"),
    (r"(?i)\b(summary|overview|highlights)\b", "summary"),
    (r"(?i)\b(report|analysis|assessment)\b", "report"),
    (r"(?i)\b(list|bullet\s+points?|enumerat[es]*)\b", "list"),
    (r"(?i)\b(table|matrix|grid)\b", "table"),
    (r"(?i)\b(yes[/ ]no|binary|go[/ ]no[- ]go)\b", "decision"),
]

# Constraint patterns
CONSTRAINT_PATTERNS = [
    r"(?i)\bmust\s+(.+?)(?:[.,;]|$)",
    r"(?i)\bshould\s+(.+?)(?:[.,;]|$)",
    r"(?i)\bwithin\s+(.+?)(?:[.,;]|$)",
    r"(?i)\blimit(?:ed)?\s+(?:to\s+)?(.+?)(?:[.,;]|$)",
    r"(?i)\bno\s+more\s+than\s+(.+?)(?:[.,;]|$)",
    r"(?i)\bat\s+(?:most|least)\s+(.+?)(?:[.,;]|$)",
]

# Sequential step patterns for pipeline detection
STEP_PATTERNS = [
    (r"(?i)\b(?:first|step\s*1)[,:]?\s*(.+?)(?:\.\s*|\n|$)", 1),
    (r"(?i)\b(?:then|next|after\s+that|step\s*2)[,:]?\s*(.+?)(?:\.\s*|\n|$)", 2),
    (r"(?i)\b(?:finally|lastly|last|step\s*3)[,:]?\s*(.+?)(?:\.\s*|\n|$)", 3),
]

# Conditional patterns
CONDITIONAL_PATTERN = r"(?i)\bif\s+(.+?)\s+then\s+(.+?)(?:\s+(?:else|otherwise)\s+(.+))?$"


class Compiler:
    """Compiles natural language into AST nodes.

    This replaces the regex-based extraction in StructuralLayer with
    proper AST output. The regex patterns are still used internally
    for NL parsing, but the output is structured.
    """

    def __init__(self, agent_codes: set[str] | None = None) -> None:
        self._agent_codes = agent_codes or set()
        for pattern in AGENT_PATTERNS:
            # Extract codes from pattern for lookup
            codes = re.findall(r"[A-Z][A-Za-z]+", pattern)
            self._agent_codes.update(codes)

    def compile(self, text: str) -> ASTNode | None:
        """Compile natural language text into an AST node.

        Returns None if the text doesn't match any recognizable pattern.
        """
        text = text.strip()
        if not text:
            return None

        # Try conditional first
        cond = self._try_conditional(text)
        if cond:
            return cond

        # Try pipeline (multi-step)
        pipeline = self._try_pipeline(text)
        if pipeline:
            return pipeline

        # Try single directive
        return self._try_directive(text)

    def _try_conditional(self, text: str) -> ConditionalNode | None:
        """Try to parse as IF...THEN...ELSE."""
        match = re.match(CONDITIONAL_PATTERN, text)
        if not match:
            return None

        condition_text = match.group(1).strip()
        then_text = match.group(2).strip()
        else_text = match.group(3).strip() if match.group(3) else None

        condition = self._parse_condition(condition_text)
        if not condition:
            return None

        then_branch = self.compile(then_text)
        if not then_branch:
            return None

        else_branch = self.compile(else_text) if else_text else None

        return ConditionalNode(
            condition=condition,
            then_branch=then_branch,
            else_branch=else_branch,
        )

    def _parse_condition(self, text: str) -> ConditionNode | None:
        """Parse a condition expression."""
        # Try field:value shorthand
        match = re.match(r"(\w+)\s*:\s*(\w+)", text)
        if match:
            return ConditionNode(
                left=ExpressionNode(field=match.group(1)),
                comparator=":",
                right=ExpressionNode(field=match.group(2)),
            )

        # Try comparator patterns
        match = re.match(r"(\w+)\s*(>=|<=|==|!=|>|<)\s*(.+)", text)
        if match:
            return ConditionNode(
                left=ExpressionNode(field=match.group(1)),
                comparator=match.group(2),
                right=ExpressionNode(field=match.group(3).strip()),
            )

        # Try "X is Y" pattern
        match = re.match(r"(?i)(\w+)\s+is\s+(\w+)", text)
        if match:
            return ConditionNode(
                left=ExpressionNode(field=match.group(1)),
                comparator=":",
                right=ExpressionNode(field=match.group(2)),
            )

        return None

    def _try_pipeline(self, text: str) -> PipelineNode | None:
        """Try to parse as a multi-step pipeline."""
        steps: list[tuple[int, str]] = []

        for pattern, order in STEP_PATTERNS:
            match = re.search(pattern, text)
            if match:
                steps.append((order, match.group(1).strip().rstrip(".,;")))

        if len(steps) < 2:
            return None

        steps.sort(key=lambda x: x[0])
        directives: list[DirectiveNode] = []

        for _, step_text in steps:
            directive = self._try_directive(step_text)
            if directive:
                directives.append(directive)
            else:
                # Create a minimal directive from the step text
                action = self._extract_action(step_text)
                target = self._extract_target(step_text, action)
                recipient = self._extract_recipient(step_text)
                directives.append(DirectiveNode(
                    action=action or "EXECUTE",
                    target=target or step_text[:50],
                    recipient=recipient,
                ))

        if len(directives) < 2:
            return None

        return PipelineNode(directives=directives)

    def _try_directive(self, text: str) -> DirectiveNode | None:
        """Try to parse as a single directive."""
        action = self._extract_action(text)
        if not action:
            return None

        recipient = self._extract_recipient(text)
        target = self._extract_target(text, action)
        params = self._extract_params(text)
        constraints = self._extract_constraints(text)
        output = self._extract_output(text)
        priority = self._extract_priority(text)
        modifiers = self._extract_modifiers(text)

        return DirectiveNode(
            action=action,
            target=target or "unspecified",
            recipient=recipient,
            params=params,
            constraints=constraints,
            output=output,
            priority=priority,
            modifiers=modifiers,
        )

    # --- Extraction helpers ---

    def _extract_action(self, text: str) -> str | None:
        for pattern, action in ACTION_MAP:
            if re.search(pattern, text):
                return action
        return None

    def _extract_recipient(self, text: str) -> RecipientNode | None:
        for pattern in AGENT_PATTERNS:
            match = re.search(pattern, text)
            if match:
                code = match.group(1)
                return RecipientNode(agent_code=code.upper())
        return None

    def _extract_target(self, text: str, action: str | None) -> str | None:
        if not action:
            return None

        # Find the action verb match and take words after it
        for pattern, act in ACTION_MAP:
            if act == action:
                match = re.search(pattern, text)
                if match:
                    after = text[match.end():].strip()
                    # Take up to the first delimiter
                    target_match = re.match(
                        r"(?:the\s+)?(.+?)(?:\s+and\s+|\s+for\s+|\s+with\s+|\s+based\s+on\s+|[.,;!?]|$)",
                        after,
                    )
                    if target_match:
                        target = target_match.group(1).strip()
                        if target and len(target) < 100:
                            return target
        return None

    def _extract_params(self, text: str) -> ParamsNode | None:
        params: list[ParamNode] = []

        # Time/period references
        time_match = re.search(
            r"(?i)\b(Q[1-4]\s*\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}|\d{4})\b",
            text,
        )
        if time_match:
            params.append(ParamNode(key="period", value=time_match.group(1)))

        # Comparison references
        compare_match = re.search(
            r"(?i)compar(?:e|ed|ing)\s+(?:to|with|against)\s+(.+?)(?:[.,;]|$)",
            text,
        )
        if compare_match:
            params.append(ParamNode(key="compare", value=compare_match.group(1).strip()))

        # Numeric values with labels
        for match in re.finditer(r"(?i)(\w+)\s+(?:of|is|are|was|=)\s+(\$?[\d,.]+%?)", text):
            params.append(ParamNode(key=match.group(1).lower(), value=match.group(2)))

        return ParamsNode(params=params) if params else None

    def _extract_constraints(self, text: str) -> ConstraintsNode | None:
        constraints: list[ConstraintNode] = []
        for pattern in CONSTRAINT_PATTERNS:
            for match in re.finditer(pattern, text):
                constraints.append(ConstraintNode(text=match.group(1).strip()))
        return ConstraintsNode(constraints=constraints) if constraints else None

    def _extract_output(self, text: str) -> OutputNode | None:
        for pattern, fmt in OUTPUT_MAP:
            if re.search(pattern, text):
                return OutputNode(format=fmt)
        return None

    def _extract_priority(self, text: str) -> PriorityNode | None:
        for pattern, level in PRIORITY_MAP:
            if re.search(pattern, text):
                return PriorityNode(level=level)
        return None

    def _extract_modifiers(self, text: str) -> list[ModifierNode]:
        mods: list[ModifierNode] = []
        seen: set[str] = set()
        for pattern, name in MODIFIER_MAP:
            if re.search(pattern, text) and name not in seen:
                mods.append(ModifierNode(name=name))
                seen.add(name)
        return mods
