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
    accuracy = "accuracy"            # single metric
    weighted = "weighted"            # weighted sum of metrics
    pareto = "pareto"               # Pareto frontier (accuracy vs efficiency)


class Budgets(BaseModel):
    max_wall_clock_minutes: int | None = None
    max_variants: int | None = None
    max_dollars: float | None = None


class ProjectSpec(BaseModel):
    name: str
    description: str = Field(..., description="Plain-language statement of what to predict.")
    task_type: TaskType
    classes: list[str] | None = None  # None for regression
    target_metric: str = "accuracy"   # e.g. accuracy, f1, auroc, mAP
    rank_objective: RankObjective = RankObjective.pareto
    metric_weights: dict[str, float] | None = None  # used when rank_objective == weighted
    data_provided: bool = False
    budgets: Budgets = Field(default_factory=Budgets)
    status: str = "defined"           # defined → data → searching → training → done
