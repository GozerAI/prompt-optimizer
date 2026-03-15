"""Grammar parsing performance benchmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from prompt_optimizer.grammar.compiler import Compiler
from prompt_optimizer.grammar.lexer import Lexer
from prompt_optimizer.grammar.parser import Parser
from prompt_optimizer.grammar.renderer import Renderer
from prompt_optimizer.grammar.validator import Validator


BENCHMARK_INPUTS = {
    "simple_directive": "@CFO ANALYZE revenue",
    "directive_with_params": "@CFO ANALYZE revenue {period=Q1-2026} -> summary !urgent ~thorough",
    "pipeline_2": "@CDO GATHER data | @CFO ANALYZE $prev",
    "parallel_block": "PAR { @CFO FORECAST revenue; @CTO ASSESS infrastructure }",
    "conditional": "IF risk:high THEN @CRO ANALYZE exposure ELSE @CFO FORECAST revenue",
    "contract": "@CFO ANALYZE revenue -> {confidence: float, recommendation: str}",
    "retry_policy": "@CFO ANALYZE revenue RETRY 3 BACKOFF exp FALLBACK @CIO",
    "nl_simple": "Please analyze the Q1 2026 revenue data.",
    "nl_complex": "Hey CFO, could you take a careful look at the Q1 2026 revenue numbers.",
}


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    input_text: str
    iterations: int
    total_ms: float
    avg_ms: float
    min_ms: float
    max_ms: float
    ops_per_second: float
    success: bool = True
    error: str | None = None


@dataclass
class BenchmarkSuite:
    """Results from a full benchmark suite run."""
    results: list[BenchmarkResult] = field(default_factory=list)
    total_ms: float = 0.0

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "total_benchmarks": len(self.results),
            "total_ms": round(self.total_ms, 2),
            "passed": sum(1 for r in self.results if r.success),
            "failed": sum(1 for r in self.results if not r.success),
            "avg_ops_per_second": (
                sum(r.ops_per_second for r in self.results if r.success)
                / max(1, sum(1 for r in self.results if r.success))
            ),
        }


def _bench(func, iterations):
    """Run func iterations times, return (total_ms, min_ms, max_ms)."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return sum(times), min(times), max(times)


class GrammarBenchmarks:
    """Benchmark suite for grammar parsing performance."""

    def __init__(self, iterations: int = 100) -> None:
        self._iterations = iterations
        self._lexer = Lexer()
        self._renderer = Renderer()
        self._compiler = Compiler()
        self._validator = Validator()

    def run_all(self) -> BenchmarkSuite:
        """Run all benchmarks."""
        suite = BenchmarkSuite()
        start = time.perf_counter()
        for name, text in BENCHMARK_INPUTS.items():
            suite.results.append(self._bench_lexer(name, text))
            if not name.startswith("nl_"):
                suite.results.append(self._bench_parser(name, text))
                suite.results.append(self._bench_roundtrip(name, text))
            else:
                suite.results.append(self._bench_compiler(name, text))
        suite.total_ms = (time.perf_counter() - start) * 1000
        return suite

    def run_lexer(self) -> BenchmarkSuite:
        """Run only lexer benchmarks."""
        suite = BenchmarkSuite()
        start = time.perf_counter()
        for name, text in BENCHMARK_INPUTS.items():
            suite.results.append(self._bench_lexer(name, text))
        suite.total_ms = (time.perf_counter() - start) * 1000
        return suite

    def _bench_lexer(self, name, text):
        try:
            total, mn, mx = _bench(lambda: self._lexer.tokenize(text), self._iterations)
            return BenchmarkResult(
                name=f"lexer/{name}", input_text=text, iterations=self._iterations,
                total_ms=total, avg_ms=total / self._iterations, min_ms=mn, max_ms=mx,
                ops_per_second=self._iterations / (total / 1000) if total > 0 else 0,
            )
        except Exception as e:
            return BenchmarkResult(
                name=f"lexer/{name}", input_text=text, iterations=0,
                total_ms=0, avg_ms=0, min_ms=0, max_ms=0, ops_per_second=0,
                success=False, error=str(e),
            )

    def _bench_parser(self, name, text):
        try:
            tokens = self._lexer.tokenize(text)
            def parse():
                Parser(list(tokens)).parse()
            total, mn, mx = _bench(parse, self._iterations)
            return BenchmarkResult(
                name=f"parser/{name}", input_text=text, iterations=self._iterations,
                total_ms=total, avg_ms=total / self._iterations, min_ms=mn, max_ms=mx,
                ops_per_second=self._iterations / (total / 1000) if total > 0 else 0,
            )
        except Exception as e:
            return BenchmarkResult(
                name=f"parser/{name}", input_text=text, iterations=0,
                total_ms=0, avg_ms=0, min_ms=0, max_ms=0, ops_per_second=0,
                success=False, error=str(e),
            )

    def _bench_roundtrip(self, name, text):
        try:
            def rt():
                tokens = self._lexer.tokenize(text)
                ast = Parser(tokens).parse()
                self._renderer.render(ast)
            total, mn, mx = _bench(rt, self._iterations)
            return BenchmarkResult(
                name=f"roundtrip/{name}", input_text=text, iterations=self._iterations,
                total_ms=total, avg_ms=total / self._iterations, min_ms=mn, max_ms=mx,
                ops_per_second=self._iterations / (total / 1000) if total > 0 else 0,
            )
        except Exception as e:
            return BenchmarkResult(
                name=f"roundtrip/{name}", input_text=text, iterations=0,
                total_ms=0, avg_ms=0, min_ms=0, max_ms=0, ops_per_second=0,
                success=False, error=str(e),
            )

    def _bench_compiler(self, name, text):
        try:
            total, mn, mx = _bench(lambda: self._compiler.compile(text), self._iterations)
            return BenchmarkResult(
                name=f"compiler/{name}", input_text=text, iterations=self._iterations,
                total_ms=total, avg_ms=total / self._iterations, min_ms=mn, max_ms=mx,
                ops_per_second=self._iterations / (total / 1000) if total > 0 else 0,
            )
        except Exception as e:
            return BenchmarkResult(
                name=f"compiler/{name}", input_text=text, iterations=0,
                total_ms=0, avg_ms=0, min_ms=0, max_ms=0, ops_per_second=0,
                success=False, error=str(e),
            )
