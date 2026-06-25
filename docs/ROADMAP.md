# Roadmap

Milestones are ordered so the **safety-critical lockbox lands before any real training**.

## M0 — Skeleton & lockbox (foundation)
- [ ] Python package scaffold, `pyproject.toml`, `ruff`, basic `cli.py`.
- [ ] Project store: create `projects/<name>/` tree, `project.yaml` schema + validation.
- [ ] Sandbox tool layer with scoped `read_file`/`write_file`/`list_dir`.
- [ ] **Lockbox enforcement** + tests proving the agent cannot read/write/traverse into `benchmark/`.
- [ ] `runs/` action logging for every tool call.

## M1 — Agent loop (no ML yet)
- [ ] `LLMClient` interface + Anthropic implementation (API-key auth, stubbed credentials boundary).
- [ ] Tool-use loop wiring the agent to the Sandbox.
- [ ] DEFINE step: agent turns a human task description into a validated `project.yaml`.
- [ ] Minimal Textual TUI: project picker + live agent feed.

## M2 — Benchmark engine
- [ ] Harness-side benchmark runner: load a variant, score on sealed holdout.
- [ ] Efficiency metrics: param count, model size, latency, throughput.
- [ ] Append-only `results.db` + ranking (single-objective first).
- [ ] `submit_for_benchmark` + `get_leaderboard` tools.
- [ ] Seal-a-holdout flow (harness splits & hides a test set).

## M3 — Training & first end-to-end project
- [ ] `run_python` sandboxed subprocess with time/memory budgets.
- [ ] PROPOSE → TRAIN → BENCHMARK working on a toy dataset (e.g. a small image classification set).
- [ ] Leaderboard in the TUI with accuracy + efficiency columns.

## M4 — Evolutionary search
- [ ] PRUNE lowest performers.
- [ ] Plateau detection + BRANCH (explore new families / mutate hyperparameters / augment).
- [ ] Multi-metric / Pareto ranking.
- [ ] Resource budget tracking (time / compute / $).

## M5 — Data sourcing
- [ ] `search_datasets` / `download_dataset` with provenance + license capture.
- [ ] Crawl/scrape fallback when the human provides no data.
- [ ] De-dup, validation, auto train/val/test split.

## M6 — Polish
- [ ] OAuth / token management UI.
- [ ] Export/portability of winning models (ONNX, etc.).
- [ ] Reproducibility bundle per model (recipe + env + data snapshot).
- [ ] Robustness benchmarks.
