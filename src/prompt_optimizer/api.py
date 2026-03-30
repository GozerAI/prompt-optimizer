"""FastAPI web API for Prompt Optimizer.

Exposes prompt compression, grammar parsing, and validation as HTTP endpoints.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from prompt_optimizer import (
    CompressedPrompt,
    CompressionContext,
    ProgressiveOptimizer,
    SchemaRegistry,
    __version__,
)
from prompt_optimizer.grammar.lexer import Lexer
from prompt_optimizer.grammar.parser import Parser
from prompt_optimizer.grammar.renderer import Renderer
from prompt_optimizer.grammar.validator import Validator

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Prompt Optimizer API",
    description="Three-layer prompt compression that translates verbose prompts into token-minimized, context-rich output.",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

_optimizer = ProgressiveOptimizer()
_registry = SchemaRegistry()


# --- Request/Response Models ---


class OptimizeRequest(BaseModel):
    text: str = Field(..., description="The prompt text to optimize")
    max_layer: int = Field(2, ge=1, le=3, description="Max compression layer (1=structural, 2=semantic, 3=context)")
    min_fidelity: float = Field(0.50, ge=0.0, le=1.0, description="Minimum fidelity score before stopping")
    target_reduction: float | None = Field(None, ge=0.0, le=1.0, description="Target token reduction ratio (e.g. 0.5 = 50%)")
    agent_codes: list[str] = Field(default_factory=list, description="Known agent codes for context")
    history: list[str] = Field(default_factory=list, description="Conversation history for deduplication")


class OptimizeResponse(BaseModel):
    original_text: str
    compressed_text: str
    layers_applied: list[int]
    tokens_original: int
    tokens_compressed: int
    reduction_pct: float
    fidelity_score: float | None
    recommendation: str | None
    drift_flags: list[dict[str, Any]]
    layer_results: list[dict[str, Any]]


class ParseRequest(BaseModel):
    source: str = Field(..., description="AIL grammar source to parse")


class ParseResponse(BaseModel):
    success: bool
    ast: dict[str, Any] | None = None
    rendered: str | None = None
    error: str | None = None


class ValidateRequest(BaseModel):
    source: str = Field(..., description="AIL grammar source to validate")


class ValidateResponse(BaseModel):
    valid: bool
    errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]


# --- Endpoints ---


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "prompt-optimizer", "version": __version__}


@app.post("/v1/optimize", response_model=OptimizeResponse)
async def optimize(req: OptimizeRequest):
    """Compress a prompt through up to 3 layers of optimization.

    - **Layer 1 (Structural)**: Strips filler, converts to typed envelopes. ~65-70% reduction.
    - **Layer 2 (Semantic)**: Deduplicates context, pipeline shorthand. ~75-80% cumulative.
    - **Layer 3 (Context)**: Replaces shared state with blackboard pointers. ~90-95% cumulative.
    """
    try:
        context = None
        if req.agent_codes or req.history:
            context = CompressionContext(
                agent_codes=req.agent_codes,
                history=req.history,
                schema_registry=_registry,
            )

        optimizer = ProgressiveOptimizer(min_fidelity=req.min_fidelity)
        result: CompressedPrompt = optimizer.optimize(
            text=req.text,
            context=context,
            target_reduction=req.target_reduction,
            max_layer=req.max_layer,
        )

        fidelity_score = None
        recommendation = None
        drift_flags = []
        if result.fidelity_report:
            fidelity_score = result.fidelity_report.overall_score
            recommendation = result.fidelity_report.recommendation.value
            drift_flags = [asdict(df) for df in result.fidelity_report.drift_flags]

        layer_results = []
        for lr in result.layer_results:
            layer_results.append({
                "layer": lr.layer,
                "input_tokens": lr.input_tokens,
                "output_tokens": lr.output_tokens,
                "reduction_pct": round(lr.reduction_pct * 100, 1),
                "risk_score": lr.risk_score,
                "transformations": lr.transformations,
            })

        return OptimizeResponse(
            original_text=result.original_text,
            compressed_text=result.compressed_text,
            layers_applied=result.layers_applied,
            tokens_original=result.token_counts.original,
            tokens_compressed=result.token_counts.compressed,
            reduction_pct=round(result.token_counts.reduction_pct * 100, 1),
            fidelity_score=fidelity_score,
            recommendation=recommendation,
            drift_flags=drift_flags,
            layer_results=layer_results,
        )
    except Exception as e:
        logger.exception("Optimization failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/parse", response_model=ParseResponse)
async def parse(req: ParseRequest):
    """Parse AIL grammar source into an AST and render back to wire format."""
    try:
        tokens = Lexer().tokenize(req.source)
        ast_node = Parser(tokens).parse()
        rendered = Renderer().render(ast_node)
        return ParseResponse(
            success=True,
            ast=_ast_to_dict(ast_node),
            rendered=rendered,
        )
    except Exception as e:
        return ParseResponse(success=False, error=str(e))


@app.post("/v1/validate", response_model=ValidateResponse)
async def validate(req: ValidateRequest):
    """Validate AIL grammar source and report errors/warnings."""
    try:
        tokens = Lexer().tokenize(req.source)
        ast_node = Parser(tokens).parse()
        result = Validator().validate(ast_node)
        return ValidateResponse(
            valid=result.is_valid if hasattr(result, "is_valid") else len(result.errors) == 0,
            errors=[asdict(e) if hasattr(e, "__dataclass_fields__") else {"message": str(e)} for e in getattr(result, "errors", [])],
            warnings=[asdict(w) if hasattr(w, "__dataclass_fields__") else {"message": str(w)} for w in getattr(result, "warnings", [])],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _ast_to_dict(node: Any) -> dict[str, Any]:
    """Convert an AST node to a serializable dict."""
    if hasattr(node, "__dataclass_fields__"):
        result = {"type": type(node).__name__}
        for f in node.__dataclass_fields__:
            val = getattr(node, f)
            if isinstance(val, list):
                result[f] = [_ast_to_dict(v) if hasattr(v, "__dataclass_fields__") else v for v in val]
            elif hasattr(val, "__dataclass_fields__"):
                result[f] = _ast_to_dict(val)
            else:
                result[f] = val
        return result
    return {"value": str(node)}
