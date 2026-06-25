# Metis

> Named for the Greek goddess of wisdom, skill, and craft — Metis is an agent harness for training **non-LLM models**.

## What we are building

Metis is a **TUI tool** that gives a frontier reasoning agent (Claude, GPT, etc. — via API token auth) the ability to autonomously **design, train, benchmark, and refine small/efficient task-specific models**. The goal: let anyone produce an efficient model for almost anything (fundus images, x-rays, flowers, dogs, audio, tabular data, …) with relative ease.

The human states *what they want to classify/predict*. The agent does the rest — sourcing data (if not provided), proposing a breadth of candidate architectures, training them, benchmarking them against a tamper-proof metric suite, pruning the weak, and branching when it hits plateaus.

Metis is **not** a tool for training LLMs. The agent *is* an LLM; the **models it produces are not** — they are compact, efficient classifiers/regressors (CNNs, gradient-boosted trees, small transformers, classical ML, etc.).

## Core principles

1. **The agent builds; the harness judges.** The agent never grades its own homework. Benchmarks are owned, executed, and recorded by the harness in a space the agent cannot read or write. This is the anti-gaming guarantee and the most important property of the system.
2. **Efficiency is a first-class metric.** We rank not just on accuracy but on parameter count, model size on disk, inference latency, and throughput. A slightly less accurate model that is 100× smaller and faster often wins.
3. **Evolutionary search.** Train many candidates, benchmark, drop the lowest performers, branch out (mutate/explore new architectures) when progress plateaus.
4. **Reproducibility.** Every model has a recorded recipe: data snapshot, code, hyperparameters, environment, and benchmark results.
5. **Human-in-the-loop, but autonomous-capable.** The TUI surfaces what the agent is doing and lets the human steer, approve data sources, or set budgets — but the agent can run the loop end-to-end.

## Per-project layout

Each project the agent works on lives in its own folder under `projects/<name>/`:

```
projects/<name>/
├── project.yaml          # task definition, target metric, budgets, status
├── data/                 # training/val/test data + labels (if labels needed)
│   ├── raw/              # as-downloaded / as-provided
│   ├── processed/        # cleaned, split, normalized
│   └── labels/           # label files / manifests
├── models/               # one subfolder per model variant tried
│   └── <variant-id>/
│       ├── recipe.yaml   # architecture, hyperparams, data snapshot ref
│       ├── train.py      # the training code the agent wrote
│       ├── weights/      # trained artifacts
│       └── card.md       # agent's notes / model card
├── benchmark/            # ⛔ LOCKED — agent cannot read or write this
│   ├── suite.py          # benchmark definitions (harness-authored/sealed)
│   ├── holdout/          # sealed test set the agent never sees
│   └── results.db        # ranked results, append-only, harness-written
└── runs/                 # logs, metrics over time, plateau detection state
```

### The benchmark lockbox (critical)

- `benchmark/` is created and sealed **by the harness**, not the agent.
- Once sealed, the agent's tool layer **blocks all read and write** to `benchmark/` (including the holdout test set). The agent cannot inspect the holdout data, cannot read the exact scoring code, and cannot edit recorded results.
- The agent submits a trained model; the **harness** runs the benchmark against the sealed holdout and writes the result to `results.db`. The agent only receives the returned scores.
- This prevents the classic failure modes: overfitting to the test set, editing the grader, or hard-coding answers.

## The agent loop

```
1. DEFINE   — Human describes the task. Agent writes project.yaml (target, classes, metric, constraints).
2. DATA     — If data provided → ingest & validate. Else → crawl/scrape/source candidate datasets,
              de-dupe, label or use provided labels, split into train/val. Harness seals a holdout
              into benchmark/ that the agent never sees.
3. PROPOSE  — Agent proposes a BREADTH of candidate model families suited to the data
              (e.g. for images: small CNN, MobileNet-class, ViT-tiny, EfficientNet, kNN baseline).
4. TRAIN    — Agent writes train.py per candidate and trains within resource budgets.
5. BENCHMARK— Harness scores each model on the sealed holdout: accuracy + efficiency metrics.
6. RANK     — Append to results.db, rank by the project's objective (multi-metric / Pareto).
7. PRUNE    — Drop the lowest performers.
8. BRANCH   — If the leaderboard plateaus, branch out: mutate top performers, try new families,
              tune hyperparameters, or augment data. Else continue refining the leaders.
9. REPEAT   — Loop 4–8 until budget exhausted or target met. Surface the leaderboard in the TUI.
```

## Benchmark metrics (default suite)

- **Accuracy / task metric** — accuracy, F1, AUROC, mAP, etc. depending on task.
- **Parameter count** — total trainable parameters.
- **Model size** — serialized size on disk (MB).
- **Inference latency** — median + p95 single-sample latency on a reference device.
- **Throughput** — samples/sec batched.
- **Robustness** (later) — performance under corruption/perturbation.

Ranking is configurable per project: single objective, weighted sum, or Pareto frontier across accuracy vs. efficiency.

## Architecture (harness)

- **Language:** Python (ML ecosystem). TUI via **Textual**.
- **Agent layer:** provider-agnostic client (Anthropic first, OpenAI later) driving a tool-use loop. Tokens via API key auth (added later; stub the auth boundary now).
- **Tool layer:** the sandbox of capabilities exposed to the agent — scoped filesystem ops, run training, request a benchmark, search/download data. **This layer enforces the `benchmark/` lockbox.**
- **Benchmark engine:** harness-side, runs sealed suites, writes append-only results.
- **Project store:** the `projects/<name>/` tree + `project.yaml` state.

```
src/metis/
├── cli.py            # entrypoint
├── tui/              # Textual app: dashboards, leaderboards, live agent feed
├── agent/            # provider-agnostic agent client + tool-use loop
├── sandbox/          # the tool layer exposed to the agent; ENFORCES benchmark lockbox
├── benchmark/        # harness-side benchmark engine + sealing
├── data_sources/     # dataset search / crawl / scrape / ingest / validate
└── projects/         # project model: project.yaml schema, lifecycle, state
```

## Build order

See `docs/ROADMAP.md`. Milestone 0 is the skeleton + lockbox enforcement (the load-bearing safety property), *before* any real training.

## Constraints & guardrails

- The agent must **never** be able to bypass the benchmark lockbox. Enforce in the tool layer, not by prompt alone.
- All agent actions are sandboxed to the active project directory.
- Resource budgets (time, compute, $) are enforced by the harness, not trusted to the agent.
- Data sourcing must respect licensing; record provenance for every dataset.

## Conventions

- Python 3.11+, type hints, `ruff` for lint/format.
- Config in YAML; results in SQLite (`results.db`).
- Keep the agent's tools small, explicit, and auditable — every tool call is logged to `runs/`.
