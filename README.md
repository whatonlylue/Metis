# Metis

> Named for the Greek goddess of wisdom, skill, and craft.

**Metis is an agent harness for training non-LLM models.** It is a terminal app
(TUI) that hands a frontier reasoning agent (Claude *or* GPT — you choose, with no
default preference) the tools to autonomously **design, train, benchmark, and
refine compact, efficient task-specific models** — for almost anything you can
classify or predict: fundus images, x-rays, flowers, birdsong, tabular records,
and more.

You say *what* you want to predict and point at your data. The agent does the
rest: it proposes a breadth of candidate architectures (small CNNs,
MobileNet-class nets, ViT-tiny, gradient-boosted trees, classical ML…), trains
them within your budget, benchmarks them on accuracy **and** efficiency, prunes
the weak, and branches out when progress plateaus.

Metis is **not** a tool for training LLMs. The agent *is* an LLM; the models it
produces are not — they are small, fast classifiers and regressors you can ship.

### The core safety property

**The agent builds the models, but the harness grades them.** Benchmarks and the
holdout test set live in a sealed lockbox the agent's tools physically cannot read
or write. So the agent can't overfit the test set, edit the grader, or hard-code
answers — the anti-gaming guarantee Metis is built around.

### What makes a model "win"

Ranking is multi-metric by default (a Pareto frontier). A model is judged on its
task metric (accuracy, F1, AUROC…) **and** its efficiency: parameter count, size
on disk, inference latency, and throughput. A slightly less accurate model that is
100× smaller and faster often wins.

### For novices and experts alike

A novice states what they want to predict, drops in data, and never has to learn
what a "holdout" or a "train/val split" is — the harness handles it. An expert can
still control splits, seeds, architectures, hyperparameters, budgets, and ranking
objectives explicitly. Sensible defaults for the novice; full control for the
expert.

See **[CLAUDE.md](./CLAUDE.md)** for the full vision and design and
**[docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)** for how the pieces fit together.
Want to contribute? See **[CONTRIBUTING.md](./CONTRIBUTING.md)**.

---

## Install

Metis needs **Python 3.11+**.

```bash
git clone <your-fork-or-this-repo> metis
cd metis
python -m venv .venv && source .venv/bin/activate

# Core harness + TUI (bundles the Anthropic SDK):
pip install -e .

# Add the providers / extras you want:
pip install -e ".[openai]"   # to drive the agent with GPT instead of Claude
pip install -e ".[ml]"       # torch / torchvision / scikit-learn for real training
```

### Pick a provider and set a key

Metis has **no default provider** — neither Claude nor GPT is preferred. You pick
one the first time you talk to the agent (press `m` in the TUI), and Metis
remembers it. Provide that provider's API key in one of two ways:

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # if you picked Claude
export OPENAI_API_KEY=sk-...            # if you picked GPT
```

…or just press `k` in the TUI and paste the key into the token manager (stored
locally, `0600`, never logged). The key is tied to the provider you chose.

---

## Quick start

### 1. Scaffold a project

```bash
metis new fundus-grading
```

This creates `projects/fundus-grading/` with a `project.yaml`, the `data/` tree,
and the sealed `benchmark/` lockbox.

### 2. Add your data

Drop your data into the project's `data/` folder. Two paths:

- **Raw, to be ingested** → put it under `data/raw/<dataset>/` and let the harness
  de-dupe, validate, and split it:

  ```bash
  metis ingest fundus-grading <dataset>
  ```

- **Already preprocessed** → drop `X.npy` / `y.npy` straight into
  `data/processed/`.

Either way, the harness **automatically carves out and seals a holdout** into the
agent-invisible `benchmark/` lockbox *before* any training can happen. You never
have to think about a test set — and the agent can never see it.

### 3. Describe the task in `project.yaml`

Open `projects/fundus-grading/project.yaml` and fill in what you want. The fields
you'll most likely touch:

```yaml
name: fundus-grading
description: Grade diabetic retinopathy severity from fundus images   # plain language
task_type: image_classification      # or tabular_classification, audio_classification, regression
classes: [none, mild, moderate, severe, proliferative]   # omit for regression
target_metric: accuracy              # accuracy, f1, auroc, mAP, …
rank_objective: pareto               # pareto | accuracy | weighted
data_provided: true

budgets:
  max_wall_clock_minutes: 30         # the harness enforces this — not the agent
  max_variants: 12
  max_dollars: 5.0

data:
  split: { train: 0.7, val: 0.15, test: 0.15 }   # test is sealed as the holdout
  split_seed: 42
```

Everything has sensible defaults — a novice can leave most of it untouched; an
expert can tune splits, budgets, prune/plateau policies, robustness corruptions,
and export settings.

### 4. Run it

```bash
metis run
```

This launches the TUI. Pick your project on the left, then talk to the agent in
the chat box ("get started", "try a smaller model", "what's winning?"). Watch the
live leaderboard fill in with accuracy + efficiency columns as candidates train
and get benchmarked. Press `m` to choose/switch the driving model, `k` to manage
your API token, `t` to change theme, `q` to quit.

> Prefer the command line? `metis --help` lists the non-interactive commands
> (`seal`, `ingest`, `benchmark`, `prune`, `budget`, `export`, `bundle`, …).
