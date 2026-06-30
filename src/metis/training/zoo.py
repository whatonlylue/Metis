"""A torch image-model zoo: prebuilt CNN families the agent can scaffold and train.

This is the image-data sibling of ``toy.py`` (which covers light sklearn families).
Torch models don't pickle like an sklearn estimator, so they carry their own
``train.py`` + ``model.py`` templates and a different inference contract — but the
``model.py`` still exposes the same ``load_model(weights_dir)`` / ``predict(model, X)``
surface the harness benchmark runner calls, so scoring is unchanged.

Data contract: image families expect ``data/processed/X.npy`` shaped ``[N, H, W, C]``
(or ``[N, H, W]`` grayscale) and integer labels in ``y.npy``. torch + torchvision must
be installed (the ``ml`` extra); training/eval of these is heavier than the sklearn
families, so callers should give ``run_python`` a generous ``memory_mb``.
"""

from __future__ import annotations

from dataclasses import dataclass

from metis.training.toy import Candidate

# model.py is static across families: it rebuilds the architecture from
# weights/meta.json and loads the trained state_dict, then predicts labels.
_MODEL_PY = '''\
"""Torch image-model inference contract used by the harness benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class _TinyCNN(nn.Module):
    """A small from-scratch CNN (no pretrained download)."""

    def __init__(self, num_classes: int, in_channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


def build_net(arch, num_classes, in_channels, pretrained=False):
    if arch == "tiny_cnn":
        return _TinyCNN(num_classes, in_channels)
    if arch == "mobilenetv3_small":
        from torchvision import models

        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_small(weights=weights)
        if in_channels != 3:
            old = net.features[0][0]
            net.features[0][0] = nn.Conv2d(
                in_channels, old.out_channels, kernel_size=old.kernel_size,
                stride=old.stride, padding=old.padding, bias=old.bias is not None,
            )
        in_feat = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_feat, num_classes)
        return net
    if arch == "resnet18":
        from torchvision import models

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        if in_channels != 3:
            old = net.conv1
            net.conv1 = nn.Conv2d(
                in_channels, old.out_channels, kernel_size=old.kernel_size,
                stride=old.stride, padding=old.padding, bias=old.bias is not None,
            )
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net
    raise ValueError(f"unknown arch: {arch}")


def _to_tensor(X):
    arr = np.asarray(X)
    if arr.ndim == 3:  # [N, H, W] grayscale -> add channel
        arr = arr[..., None]
    arr = arr.astype("float32")
    if arr.size and arr.max() > 1.0:
        arr = arr / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # NCHW


def load_model(weights_dir):
    weights_dir = Path(weights_dir)
    meta = json.loads((weights_dir / "meta.json").read_text())
    net = build_net(meta["arch"], meta["num_classes"], meta["in_channels"], pretrained=False)
    net.load_state_dict(torch.load(weights_dir / "model.pt", map_location="cpu"))
    net.eval()
    return net


def predict(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(_to_tensor(X))
    return logits.argmax(dim=1).cpu().numpy()
'''

# train.py template. Tokens are substituted via .replace (not .format) to avoid
# escaping every brace in the embedded torch code.
_TRAIN_PY = '''\
"""PROPOSE/TRAIN candidate (torch image model): __FAMILY__."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

variant_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(variant_dir))  # make sibling model.py importable under run_python
from model import build_net  # noqa: E402  (shares the architecture definition)

project_root = variant_dir.parents[1]
processed = project_root / "data" / "processed"

X = np.load(processed / "X.npy")
if X.ndim == 3:
    X = X[..., None]
X = X.astype("float32")
if X.size and X.max() > 1.0:
    X = X / 255.0
y = np.load(processed / "y.npy").astype("int64")

arch = "__ARCH__"
hp = __HPARAMS__
num_classes = int(y.max()) + 1
in_channels = X.shape[-1]

net = build_net(arch, num_classes, in_channels, pretrained=bool(hp.get("pretrained", False)))

Xt = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()
yt = torch.from_numpy(y)
loader = DataLoader(
    TensorDataset(Xt, yt), batch_size=int(hp.get("batch_size", 32)), shuffle=True
)

opt = torch.optim.Adam(net.parameters(), lr=float(hp.get("lr", 1e-3)))
loss_fn = nn.CrossEntropyLoss()
net.train()
for _epoch in range(int(hp.get("epochs", 5))):
    for xb, yb in loader:
        opt.zero_grad()
        loss_fn(net(xb), yb).backward()
        opt.step()

weights = variant_dir / "weights"
weights.mkdir(parents=True, exist_ok=True)
torch.save(net.state_dict(), weights / "model.pt")
(weights / "meta.json").write_text(
    json.dumps({"arch": arch, "num_classes": num_classes, "in_channels": in_channels})
)

param_count = int(sum(p.numel() for p in net.parameters()))
(variant_dir / "recipe.yaml").write_text(
    yaml.safe_dump({"architecture": "__FAMILY__", "param_count": param_count}, sort_keys=False)
)
print(f"trained __FAMILY__: {param_count} params on {len(X)} samples")
'''


@dataclass(frozen=True)
class TorchFamilySpec:
    """A torch image model family + its tunable hyperparameters.

    ``arch`` is the key ``model.py:build_net`` dispatches on. ``needs_download`` flags
    families that fetch pretrained weights from torchvision (network + extra memory).
    """

    key: str
    family: str
    arch: str
    description: str
    default_hparams: dict[str, object]
    hparam_grid: dict[str, list[object]]
    needs_download: bool = False
    min_memory_mb: int = 2048


TORCH_FAMILIES: dict[str, TorchFamilySpec] = {
    "tiny_cnn": TorchFamilySpec(
        key="tiny_cnn",
        family="tiny_cnn",
        arch="tiny_cnn",
        description="Small from-scratch CNN (2 conv blocks). No download; fast, very compact.",
        default_hparams={"epochs": 5, "lr": 1e-3, "batch_size": 32},
        hparam_grid={"epochs": [3, 5, 10], "lr": [1e-3, 3e-4], "batch_size": [16, 32, 64]},
        needs_download=False,
        min_memory_mb=2048,
    ),
    "mobilenetv3_small": TorchFamilySpec(
        key="mobilenetv3_small",
        family="mobilenetv3_small",
        arch="mobilenetv3_small",
        description="MobileNetV3-Small (torchvision); efficient, optionally pretrained → fine-tune.",
        default_hparams={"epochs": 5, "lr": 1e-3, "batch_size": 32, "pretrained": True},
        hparam_grid={"epochs": [3, 5, 10], "lr": [1e-3, 3e-4], "pretrained": [True, False]},
        needs_download=True,
        min_memory_mb=4096,
    ),
    "resnet18": TorchFamilySpec(
        key="resnet18",
        family="resnet18",
        arch="resnet18",
        description="ResNet18 (torchvision); stronger accuracy, larger, optionally pretrained.",
        default_hparams={"epochs": 5, "lr": 1e-3, "batch_size": 32, "pretrained": True},
        hparam_grid={"epochs": [3, 5, 10], "lr": [1e-3, 3e-4], "pretrained": [True, False]},
        needs_download=True,
        min_memory_mb=4096,
    ),
}


def build_torch_candidate(
    spec: TorchFamilySpec,
    hparams: dict[str, object],
    variant_id: str,
) -> Candidate:
    """Materialize a concrete torch training candidate from a family + hyperparameters."""
    train_py = (
        _TRAIN_PY.replace("__FAMILY__", spec.family)
        .replace("__ARCH__", spec.arch)
        .replace("__HPARAMS__", repr(dict(hparams)))
    )
    return Candidate(
        variant_id=variant_id,
        family=spec.family,
        train_py=train_py,
        model_py=_MODEL_PY,
    )
