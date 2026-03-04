# xrag — Subjective Logic for Retrieval-Augmented Generation

> Beyond Scalar Confidence: Subjective Logic as a Unified Uncertainty Interface for Retrieval-Augmented Generation

## Overview

**xrag** replaces the disconnected, ad-hoc scalar confidence heuristics across RAG pipeline decisions with Subjective Logic's opinion algebra — a mathematically grounded framework for evidence estimation, fusion, conflict detection, temporal decay, trust discounting, multi-hop deduction, and calibrated abstention.

## Key Contributions

1. **SL-RAG Pipeline** — A modular uncertainty layer using 18+ SL operators between retrieval and generation
2. **UQ-RAGBench** — First evaluation harness jointly measuring RAG quality and UQ calibration under controlled corruption
3. **Empirical Analysis** — Head-to-head comparison against softmax entropy, semantic entropy, conformal prediction, and CRAG-style ternary confidence across 4 benchmarks

## Benchmarks

| Benchmark | Type | Purpose |
|-----------|------|---------|
| Natural Questions | Single-hop QA | Standard baseline |
| PopQA | Long-tail entity QA | Knowledge boundary testing |
| HotpotQA | 2-hop multi-hop QA | Deduction operator validation |
| MuSiQue | 2–4 hop compositional QA | Deep deduction chains |

## Installation

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/xrag.git
cd xrag

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest
```

## Project Structure

```
xrag/
├── src/xrag/
│   ├── opinion_estimation/   # NLI and LLM-as-judge → SL opinions
│   ├── pipeline/             # SL-RAG pipeline layers
│   ├── baselines/            # UQ baseline implementations
│   ├── benchmarks/           # Data loaders and retrieval
│   ├── evaluation/           # Metrics and evaluation harness
│   └── utils/                # Configuration and helpers
├── tests/                    # Test suites (TDD)
├── experiments/              # Configs, scripts, results
├── paper/                    # LaTeX source, figures, tables
└── docs/                     # Execution plan and documentation
```

## Dependencies

Core SL operators are provided by [`jsonld-ex`](https://pypi.org/project/jsonld-ex/), which implements:
- Opinion algebra (cumulative/averaging fusion, trust discount, deduction)
- Byzantine-resistant fusion (3 strategies, 4 distance metrics)
- Temporal decay (3 decay functions, temporal fusion pipelines)
- Conflict detection (pairwise conflict, cohesion scoring)

## License

MIT

## Citation

Paper forthcoming.
