"""Schema for ``project.yaml`` — the task definition the agent works against."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


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


class SplitRatios(BaseModel):
    """Train/val/test split fractions. Must sum to 1.0; test is sealed as the holdout."""

    train: float = Field(default=0.7, gt=0.0, lt=1.0)
    val: float = Field(default=0.15, ge=0.0, lt=1.0)
    test: float = Field(default=0.15, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _sum_to_one(self) -> SplitRatios:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train+val+test must sum to 1.0, got {total}")
        return self


class LicensePolicySpec(BaseModel):
    """Policy for accepting dataset licenses when sourcing data.

    ``require_license`` refuses datasets with an unknown/missing license outright;
    set it False to keep-but-flag instead. ``allowed_licenses`` (if set) restricts
    acceptance to those license identifiers.
    """

    require_license: bool = True
    allowed_licenses: list[str] | None = None


class DataSourceRef(BaseModel):
    """Provenance reference recorded in project.yaml for each fetched dataset."""

    dataset: str
    source: str  # provider name
    identifier: str = ""
    url: str | None = None
    license: str | None = None
    license_ok: bool = True
    checksum: str | None = None
    retrieved_at: str | None = None


class DataConfig(BaseModel):
    """DATA-step configuration + recorded provenance for the project."""

    split: SplitRatios = Field(default_factory=SplitRatios)
    split_seed: int = 42
    license_policy: LicensePolicySpec = Field(default_factory=LicensePolicySpec)
    sources: list[DataSourceRef] = Field(default_factory=list)


class CorruptionSpec(BaseModel):
    """One robustness corruption applied (harness-side) to the holdout features.

    ``name`` selects the perturbation ("gaussian_noise", "feature_dropout",
    "scaling"); ``severity`` scales its strength (std multiplier, dropout
    probability, or scale offset respectively).
    """

    name: str
    severity: float = Field(default=0.2, ge=0.0)


def _default_corruptions() -> list[CorruptionSpec]:
    return [
        CorruptionSpec(name="gaussian_noise", severity=0.5),
        CorruptionSpec(name="feature_dropout", severity=0.2),
        CorruptionSpec(name="scaling", severity=0.2),
    ]


class RobustnessConfig(BaseModel):
    """Robustness-benchmark settings. Evaluated HARNESS-side on the sealed holdout.

    The agent never sees the holdout or the perturbations; the runner perturbs
    features inside the trusted process and scores against labels it keeps.
    """

    seed: int = 0
    corruptions: list[CorruptionSpec] = Field(default_factory=_default_corruptions)


class ExportConfig(BaseModel):
    """Portability/export settings for winning models.

    ``prefer_onnx`` requests an ONNX artifact when ``onnx``/``skl2onnx`` are
    installed; otherwise the harness falls back to a self-contained pickle
    bundle.
    """

    prefer_onnx: bool = True


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
    data: DataConfig = Field(default_factory=DataConfig)
    robustness: RobustnessConfig = Field(default_factory=RobustnessConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    status: str = "defined"  # defined → data → searching → training → done
