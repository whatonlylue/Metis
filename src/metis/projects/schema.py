"""Schema for ``project.yaml`` — the task definition the agent works against."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    image_classification = "image_classification"
    tabular_classification = "tabular_classification"
    audio_classification = "audio_classification"
    regression = "regression"
    # extend as new modalities are supported


class RankObjective(str, Enum):
    accuracy = "accuracy"  # single metric
    weighted = "weighted"  # weighted sum of metrics
    pareto = "pareto"  # Pareto frontier (accuracy vs efficiency)


class Budgets(BaseModel):
    max_wall_clock_minutes: float | None = None
    max_variants: int | None = None
    max_dollars: float | None = None
    # Simple, configurable cost model: dollars accrued per wall-clock minute of
    # harness-run training. 0.0 (default) means $ tracking is informational only.
    dollars_per_minute: float = 0.0


class PrunePolicy(BaseModel):
    """How the harness prunes weak variants. Pruning marks (reversible), never deletes.

    If ``keep_top_k`` is set it wins; otherwise ``drop_bottom_fraction`` is used;
    if neither is set, pruning is a no-op (nothing is dropped).
    """

    keep_top_k: int | None = None
    drop_bottom_fraction: float | None = Field(default=None, ge=0.0, le=1.0)


class PlateauPolicy(BaseModel):
    """Plateau detection: stop improving when the best objective stalls.

    A plateau is declared when the best objective over the last ``window``
    benchmark rounds improved by no more than ``epsilon``.
    """

    epsilon: float = Field(default=1e-3, ge=0.0)
    window: int = Field(default=3, ge=1)


class ProjectSpec(BaseModel):
    name: str
    description: str = Field(..., description="Plain-language statement of what to predict.")
    task_type: TaskType
    classes: list[str] | None = None  # None for regression
    target_metric: str = "accuracy"  # e.g. accuracy, f1, auroc, mAP
    rank_objective: RankObjective = RankObjective.pareto
    metric_weights: dict[str, float] | None = None  # used when rank_objective == weighted
    data_provided: bool = False
    budgets: Budgets = Field(default_factory=Budgets)
    prune_policy: PrunePolicy = Field(default_factory=PrunePolicy)
    plateau: PlateauPolicy = Field(default_factory=PlateauPolicy)
    status: str = "defined"  # defined → data → searching → training → done
