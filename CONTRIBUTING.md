# Contributing to Metis

Thanks for wanting to help build Metis. This guide covers getting a development
environment running, the conventions we follow, and how to land a change.

## Development setup

Metis needs **Python 3.11+**.

```bash
# 1. Fork on GitHub, then clone your fork
git clone https://github.com/<you>/metis
cd metis

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install Metis in editable mode with the dev + optional extras
pip install -e ".[dev,ml]"
```

By default Metis stores all projects and settings under `~/.metis/`. While
hacking, export `METIS_HOME` to keep a development instance's data isolated from
your real one (and out of the repo):

```bash
export METIS_HOME="$PWD/.metis-dev"   # projects + settings land here instead
```

The extras:

- `dev` — `ruff` (lint/format) and `pytest`.
- `ml` — `torch` / `torchvision` / `scikit-learn` / `numpy` for the training and
  benchmarking paths.

The agent talks to every provider through `litellm` (a core dependency), so there
is no per-provider SDK extra to install.

## Running checks

```bash
ruff check .          # lint
ruff format .         # auto-format
pytest                # the full test suite
pytest tests/test_providers.py -q     # a single file, while iterating
```

Please run `ruff check .` and the relevant tests before opening a PR. New behavior
should come with tests — the existing files under `tests/` are good templates.

## Conventions

- **Python 3.11+, full type hints.** `ruff` is the single source of truth for
  lint and formatting (line length 100, see `pyproject.toml`).
- **Config in YAML, results in SQLite.** Project state lives in `project.yaml`;
  benchmark results in an append-only `results.db`.
- **Keep the agent's tools small, explicit, and auditable.** Every tool call is
  logged to `runs/`.
- **No provider preference.** Metis is neutral between agent providers (Anthropic,
  OpenAI, Gemini, …). Never hard-code or default to one; the user always picks.
  Providers are driven through `litellm` (`src/metis/agent/litellm_client.py`) via
  a free-form model string, so any litellm-supported provider works with no change
  to the loop, tools, or session.
- **Never weaken the benchmark lockbox.** The agent's tool layer must continue to
  block all reads and writes to `benchmark/`. This is enforced in code, not by
  prompt. Any change near the sandbox needs lockbox tests to stay green.

## Landing a change

1. Branch off `main` (e.g. `git checkout -b feat/ollama-provider`).
2. Make your change with tests and docs.
3. Run `ruff check .` and `pytest`.
4. Open a PR describing the *why*, not just the *what*. Link any relevant roadmap
   item below.

## Architecture orientation

See **[docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)** for how the harness fits
together, and **[CLAUDE.md](./CLAUDE.md)** for the project vision, per-project
layout, and the agent loop. The package map:

```
src/metis/
├── cli.py            # entrypoint
├── tui/              # Textual app: dashboards, leaderboards, live agent feed
├── agent/            # provider-agnostic agent client + tool-use loop
├── sandbox/          # the tool layer exposed to the agent; ENFORCES the lockbox
├── benchmark/        # harness-side benchmark engine + sealing
├── data_sources/     # dataset ingest / validate / provenance
└── projects/         # project model: project.yaml schema, lifecycle, state
```

---

## Roadmap — features we'd love help with

The core build (skeleton + lockbox, agent loop, benchmark engine, evolutionary
search, data ingestion, export/reproducibility, token-cost ergonomics) is in
place. These are the directions we want to take Metis next. Pick one, open an
issue to claim it, and have at it.

### Agent providers & models
- [x] **More cloud providers** — Gemini, Mistral, Cohere, … — now work for free via
      `litellm`: pick any litellm model string, no registry entry or loop change.
- [x] **Local / self-hosted models** as agent drivers — **Ollama**, **LM Studio**,
      **vLLM**, **llama.cpp** — reachable through their `litellm` model strings
      (e.g. `ollama/llama3`), so the harness can run offline with no cloud key.
      *(Follow-up: a friendlier picker/onboarding for local endpoints.)*
- [ ] **Per-project model overrides** — let a project pin which agent model drives
      it, independent of the global pick.

### Chat box power-ups
- [ ] **Slash commands in the chat box** — `/leaderboard`, `/prune`, `/budget`,
      `/export <variant>`, `/model`, `/seal`, `/help` — fast actions without
      leaving the keyboard.
- [ ] **Shell commands from the chat box** — a `!<command>` prefix to run a
      sandboxed shell command and stream its output into the feed, for power users
      inspecting their own data/artifacts.
- [ ] **Command palette / autocomplete** for projects, variants, and commands.

### Modalities & search
- [ ] **More task types** — object detection, segmentation, text/NLP
      classification, and time-series forecasting.
- [ ] **Smarter hyperparameter search** — Bayesian / Optuna-style tuning as a
      branch strategy.
- [ ] **Opt-in data sourcing** — bring back guarded dataset search/crawl so a
      novice with no data can still get started (license-respecting, provenance
      recorded).

### Training, export & ops
- [ ] **Remote / GPU training backends** — offload training to a GPU runner while
      the TUI stays local.
- [ ] **More export targets** — CoreML, TFLite, and GGUF for edge/mobile
      deployment alongside ONNX.
- [ ] **Run comparison & diffing** — compare two variants' recipes and metrics
      side by side; one-command re-run from a reproducibility bundle.
- [ ] **A companion web UI** sharing the same harness core as the TUI.

### Fun Stuff
- [ ] **Needs** a cool brand logo
Have an idea that isn't here? Open an issue — we'd love to hear it.
