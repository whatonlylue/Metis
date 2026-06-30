"""Tests for the agent-facing model-template tools (list/instantiate).

These let the agent scaffold a proven model family instead of hand-writing
train.py/model.py — cutting the token + retry cost that previously crashed a
hand-written torch model (no torch installed). The sklearn path is exercised
end-to-end; the torch path is scaffolded and only run when torch is present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from metis.agent.tools import build_template_tools  # noqa: E402
from metis.projects import create_project  # noqa: E402
from metis.projects.schema import ProjectSpec, TaskType  # noqa: E402
from metis.sandbox import run_python  # noqa: E402
from metis.training import FAMILIES, TORCH_FAMILIES, prepare_toy_dataset  # noqa: E402


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    spec = ProjectSpec(
        name="digits",
        description="classify 8x8 handwritten digits",
        task_type=TaskType.image_classification,
        classes=[str(d) for d in range(10)],
        target_metric="accuracy",
    )
    return create_project(tmp_path / "digits", spec)


def _tool(tools, name: str):
    return next(t for t in tools if t.name == name)


def test_list_model_templates_includes_all_families(project_root: Path) -> None:
    out = _tool(build_template_tools(project_root), "list_model_templates").handler({})
    for key in FAMILIES:
        assert key in out
    for key in TORCH_FAMILIES:
        assert key in out


def test_instantiate_sklearn_template_scaffolds_and_runs(project_root: Path) -> None:
    prepare_toy_dataset(project_root)
    tools = build_template_tools(project_root)
    msg = _tool(tools, "instantiate_template").handler(
        {"template": "logreg", "variant_id": "lr1", "hparams": {"C": 0.5}}
    )
    assert "scaffolded" in msg
    variant = project_root / "models" / "lr1"
    assert (variant / "train.py").exists()
    assert (variant / "model.py").exists()

    result = run_python(project_root, "models/lr1/train.py", timeout_s=120)
    assert result.exit_code == 0, result.stderr
    assert (variant / "weights" / "model.pkl").exists()
    assert (variant / "recipe.yaml").exists()


def test_instantiate_unknown_template_errors(project_root: Path) -> None:
    msg = _tool(build_template_tools(project_root), "instantiate_template").handler(
        {"template": "does_not_exist", "variant_id": "x"}
    )
    assert msg.startswith("error: unknown template")


def test_instantiate_torch_template_scaffolds(project_root: Path) -> None:
    tools = build_template_tools(project_root)
    msg = _tool(tools, "instantiate_template").handler(
        {"template": "tiny_cnn", "variant_id": "cnn1"}
    )
    assert "scaffolded" in msg
    variant = project_root / "models" / "cnn1"
    train_src = (variant / "train.py").read_text()
    assert "build_net" in train_src and "tiny_cnn" in train_src
    assert (variant / "model.py").exists()


def test_torch_tiny_cnn_trains_when_torch_available(project_root: Path) -> None:
    pytest.importorskip("torch")
    import numpy as np

    # Tiny synthetic image set: [N, H, W, C] uint8 + integer labels.
    processed = project_root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    X = rng.integers(0, 255, size=(24, 8, 8, 3), dtype=np.uint8)
    y = rng.integers(0, 3, size=(24,), dtype=np.int64)
    np.save(processed / "X.npy", X)
    np.save(processed / "y.npy", y)

    tools = build_template_tools(project_root)
    _tool(tools, "instantiate_template").handler(
        {"template": "tiny_cnn", "variant_id": "cnn1", "hparams": {"epochs": 1, "batch_size": 8}}
    )
    result = run_python(project_root, "models/cnn1/train.py", timeout_s=300, memory_mb=4096)
    assert result.exit_code == 0, result.stderr
    assert (project_root / "models" / "cnn1" / "weights" / "model.pt").exists()
