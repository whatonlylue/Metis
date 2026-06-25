# Metis

An agent harness for training **non-LLM models**. Named for the Greek goddess of wisdom and craft.

Metis is a TUI tool that lets a reasoning agent (Claude, GPT) autonomously design, train, benchmark, and refine compact task-specific models — for almost anything you can classify or predict. You say *what* you want; the agent sources data, proposes a breadth of architectures, trains them, and ranks them on a **tamper-proof** benchmark of accuracy **and** efficiency.

The key safety property: **the agent builds the models, but the harness grades them.** Benchmarks live in a sealed lockbox the agent cannot read or write — so it can't overfit the test set or game the grader.

See **[CLAUDE.md](./CLAUDE.md)** for the full vision and design, **[docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)** for how it fits together, and **[docs/ROADMAP.md](./docs/ROADMAP.md)** for the build plan.

## Status

Early scaffolding. Milestone M0 (skeleton + lockbox) is the first build target.

## Quickstart (planned)

```bash
pip install -e .
metis new fundus-classifier      # scaffold a project
metis run fundus-classifier      # launch the TUI + agent loop
```
