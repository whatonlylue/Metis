"""Project store: scaffold ``projects/<name>/`` trees and validate ``project.yaml``."""

from __future__ import annotations

from pathlib import Path

import yaml

from metis.projects.schema import ProjectSpec

SUBDIRS = [
    "data/raw",
    "data/processed",
    "data/labels",
    "models",
    "benchmark/holdout",
    "runs",
]


def create_project(root: Path, spec: ProjectSpec) -> Path:
    """Scaffold the project directory tree and write a validated ``project.yaml``.

    Raises ``FileExistsError`` if ``root`` already exists.
    """
    if root.exists():
        raise FileExistsError(f"Project already exists: {root}")
    for sub in SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    write_project_yaml(root, spec)
    return root


# Inline help for fields whose right value isn't obvious from the field name alone.
# Keyed by field name; nested fields (currently just under `budgets:`) are matched
# regardless of indent. Fields like `name`/`description`/`status` are skipped because
# their expected value is self-evident.
FIELD_COMMENTS: dict[str, str] = {
    "classes": 'List of class labels, e.g. ["daisy", "dandelion", "rose"]. Leave null for regression.',
    "target_metric": 'Metric to optimize: "accuracy", "f1", "auroc", "mAP" for classification, '
    'or "rmse", "mae" for regression.',
    "rank_objective": '"accuracy" (rank by target_metric alone), "weighted" (weighted sum of '
    'metrics, see metric_weights), or "pareto" (accuracy vs. efficiency frontier).',
    "metric_weights": 'Only used when rank_objective is "weighted", '
    "e.g. {accuracy: 0.7, latency_ms: 0.3}. Leave null otherwise.",
    "max_wall_clock_minutes": "Training time budget in minutes, e.g. 120. Leave null for no limit.",
    "max_variants": "Max number of model variants to train, e.g. 20. Leave null for no limit.",
    "max_dollars": "Max spend in USD across training + agent calls, e.g. 5.0. Leave null for no limit.",
    "dollars_per_minute": "Cost model: USD charged per wall-clock minute of training, e.g. 0.05. "
    "0 means $ tracking is informational only.",
    "keep_top_k": "Prune policy: keep only the top-k ranked variants, e.g. 5. Takes precedence "
    "over drop_bottom_fraction. Leave null to use the fraction instead.",
    "drop_bottom_fraction": "Prune policy: drop the worst fraction of variants, e.g. 0.5. "
    "Used only when keep_top_k is null.",
    "epsilon": "Plateau detection: minimum improvement in the best objective over `window` "
    "rounds to count as progress, e.g. 0.001.",
    "window": "Plateau detection: number of recent benchmark rounds to look back over, e.g. 3.",
}


def write_project_yaml(root: Path, spec: ProjectSpec) -> None:
    """Serialize ``spec`` to ``<root>/project.yaml``, annotated with inline help.

    Comments are inserted above fields listed in ``FIELD_COMMENTS``; they don't
    affect parsing, so ``load_project`` round-trips the result unchanged.
    """
    dumped = yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False)
    (root / "project.yaml").write_text(_annotate(dumped))


def _annotate(dumped_yaml: str) -> str:
    annotated_lines: list[str] = []
    for line in dumped_yaml.splitlines():
        key = line.strip().split(":", 1)[0]
        comment = FIELD_COMMENTS.get(key)
        if comment:
            indent = line[: len(line) - len(line.lstrip(" "))]
            annotated_lines.append(f"{indent}# {comment}")
        annotated_lines.append(line)
    return "\n".join(annotated_lines) + "\n"


def load_project(root: Path) -> ProjectSpec:
    """Read and validate ``<root>/project.yaml``."""
    data = yaml.safe_load((root / "project.yaml").read_text())
    return ProjectSpec.model_validate(data)
