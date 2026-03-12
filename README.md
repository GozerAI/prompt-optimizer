# Prompt Optimizer

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Three-layer prompt compression and orchestration grammar library. Reduce token usage by up to 95% while preserving prompt fidelity through progressive optimization layers.

Part of the [GozerAI](https://gozerai.com) ecosystem.

## Features

- **Three compression layers** with increasing reduction and configurable risk tolerance
- **Progressive optimization** that applies layers incrementally with fidelity escape hatches
- **Full orchestration grammar** with lexer, parser, renderer, validator, and compiler
- **Runtime executor** for orchestration programs with contracts and context management
- **CLI tool** for parsing, optimizing, and running prompts
- **Zero runtime dependencies** (tiktoken optional for token counting)

## Installation

```bash
pip install -e .

# With token counting support
pip install -e ".[tiktoken]"

# With LLM-based fidelity judging
pip install -e ".[llm-judge]"
```

## Compression Layers

| Layer | Name | Reduction | Risk | Description |
|-------|------|-----------|------|-------------|
| L1 | Structural | ~65-70% | ~2% | Filler stripping and typed envelope wrapping |
| L2 | Semantic | ~80% | ~10% | Context deduplication, pipeline shorthand, schema abbreviations |
| L3 | Context | ~95% | ~20% | Blackboard pointers for shared state (`bb:ns:key@vN`) |

## Quick Start

### As a Library

```python
from prompt_optimizer import ProgressiveOptimizer

optimizer = ProgressiveOptimizer()

original = """
You are a helpful assistant. Please analyze the following data
and provide a comprehensive summary. Make sure to include all
relevant details and format the output clearly.
"""

# Optimize with automatic layer selection
result = optimizer.optimize(original)
print(result.text)        # Compressed prompt
print(result.layer)       # Layer used (L1, L2, or L3)
print(result.reduction)   # Compression ratio
```

### Layer-by-Layer Control

```python
from prompt_optimizer.layers import L1Structural, L2Semantic, L3Context

l1 = L1Structural()
result = l1.compress("Your verbose prompt here...")
print(result.text)       # Filler stripped, envelope wrapped
print(result.fidelity)   # Fidelity score (0.0 - 1.0)
```

### Orchestration Grammar

```python
from prompt_optimizer.grammar import Lexer, Parser, Renderer, Validator

source = """
PAR {
  agent_a: "Analyze revenue data"
  agent_b: "Review customer feedback"
}
SEQ {
  summarizer: "Combine results from $prev"
}
"""

tokens = Lexer().tokenize(source)
ast = Parser().parse(tokens)
errors = Validator().validate(ast)
output = Renderer().render(ast)
```

### Runtime Execution

```python
from prompt_optimizer.runtime import Executor, ExecutionContext

executor = Executor()
context = ExecutionContext()

# Execute an orchestration program
result = await executor.execute(ast, context)
```

## CLI Reference

```bash
# Parse and display AST
po parse program.ail

# Emit rendered output
po emit program.ail

# Validate syntax
po validate program.ail

# Run an orchestration program
po run program.ail

# Optimize a prompt
po optimize "Your prompt text here"

# Interactive REPL
po repl

# Process a file
po file input.txt --layer l2 --output compressed.txt
```

## Orchestration Features

- **PAR/SEQ blocks** for parallel and sequential agent coordination
- **Pipeline references** with `$prev` for chaining outputs
- **Blackboard pointers** (`bb:namespace:key@vN`) for shared state
- **RETRY/BACKOFF/FALLBACK** for resilient execution
- **Response contracts** for output validation
- **Compiler** for AST-to-executable transformation

## Architecture

```
prompt_optimizer/
  layers/          # L1, L2, L3 compression layers
  grammar/         # Lexer, Parser, AST nodes, Renderer, Validator, Compiler
  runtime/         # Executor, ExecutionContext, Contracts
  progressive.py   # ProgressiveOptimizer (automatic layer selection)
  blackboard.py    # Shared state management
  envelope.py      # TypedEnvelope for structured prompts
  fidelity.py      # Fidelity scoring
  cli.py           # CLI entry point
```

## License

MIT — see [LICENSE](LICENSE) for details. Learn more at [gozerai.com](https://gozerai.com).
