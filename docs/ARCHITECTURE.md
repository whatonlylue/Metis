# Architecture

This document goes one level deeper than `CLAUDE.md` on *how* the pieces fit.

## Component map

```
┌─────────────────────────────────────────────────────────────┐
│                          TUI (Textual)                       │
│   leaderboard · live agent feed · project picker · controls  │
└───────────────┬─────────────────────────────────────────────┘
                │ observes / steers
┌───────────────▼─────────────────────────────────────────────┐
│                       Orchestrator                            │
│   drives the DEFINE→DATA→PROPOSE→TRAIN→BENCH→RANK→PRUNE loop  │
└───┬───────────────┬───────────────────┬─────────────────┬────┘
    │               │                   │                 │
┌───▼────┐   ┌──────▼──────┐    ┌───────▼──────┐   ┌──────▼──────┐
│ Agent  │   │  Sandbox    │    │  Benchmark   │   │   Project   │
│ client │   │ (tool layer)│    │   engine     │   │   store     │
│        │   │  ENFORCES   │    │  (sealed)    │   │ project.yaml│
│ tool   │◄─►│  lockbox    │    │              │   │  + tree     │
│ loop   │   └─────────────┘    └──────────────┘   └─────────────┘
└────────┘
```

## Trust boundary

The single most important boundary: **the agent only touches the world through the Sandbox tool layer.** It never gets raw filesystem or shell access. Every capability is a named tool with explicit, validated arguments, and every call is logged.

### Lockbox enforcement

`benchmark/` (suite code + sealed holdout + results.db) is off-limits to the agent. Enforcement lives in the Sandbox, e.g.:

- `read_file(path)` / `write_file(path)` / `list_dir(path)` resolve the path, then **reject** anything that resolves inside `<project>/benchmark/`.
- Path resolution is canonicalized (resolve symlinks, `..`) to prevent traversal escapes.
- The agent's only interaction with benchmarks is `submit_for_benchmark(model_variant_id)` → returns scores. It cannot read the holdout, the suite source, or edit results.
- The harness writes `results.db` append-only; the agent has no write tool that targets it.

## Agent tool surface (initial)

| Tool | Purpose |
|------|---------|
| `read_file` / `write_file` / `list_dir` | scoped to project, minus `benchmark/` |
| `run_python` | execute a training/eval script in a sandboxed subprocess with budgets |
| `search_datasets` / `download_dataset` | find & fetch candidate data (provenance recorded) |
| `submit_for_benchmark` | hand a trained variant to the harness; get back scores |
| `get_leaderboard` | read current ranked results (scores only, harness-served) |
| `record_note` | write to a variant's `card.md` |

## Data flow for a model variant

1. Agent writes `models/<id>/train.py` + `recipe.yaml` via `write_file`.
2. Agent calls `run_python` to train → weights land in `models/<id>/weights/`.
3. Agent calls `submit_for_benchmark("<id>")`.
4. Harness loads the weights, runs `benchmark/suite.py` against `benchmark/holdout/`, measures accuracy + efficiency, appends a row to `results.db`.
5. Harness returns the scores to the agent and updates the TUI leaderboard.

## Plateau detection & branching

The orchestrator tracks the best objective value over the last *k* benchmarked variants. If improvement < ε over a window, it signals **plateau** to the agent and biases the next PROPOSE step toward exploration (new model families, data augmentation, hyperparameter mutation) rather than exploitation.

## Provider abstraction

`agent/` exposes a thin `LLMClient` interface (`chat`, `tool_loop`). Anthropic is the first implementation. Auth is API-key based; for now stub a `Credentials` boundary so swapping providers / adding OAuth later is localized.

## Why these tech choices

- **Python** — non-negotiable for the ML training ecosystem (PyTorch, scikit-learn, timm, etc.).
- **Textual** — modern async Python TUI, good for live dashboards and leaderboards.
- **SQLite** — zero-config, append-only-friendly, easy to seal and query for results.
- **YAML** — human-editable project/recipe config.
- **Subprocess isolation** for `run_python` — keeps training crashes/timeouts/leaks away from the harness and lets us enforce budgets.
