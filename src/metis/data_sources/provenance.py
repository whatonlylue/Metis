"""Provenance + license capture for sourced datasets.

Every dataset fetched through a :class:`~metis.data_sources.provider.DatasetProvider`
records *where it came from* and *under what license* before it is allowed into a
project. This is the licensing guardrail from CLAUDE.md ("Data sourcing must respect
licensing; record provenance for every dataset") enforced in code, not by prompt.

A :class:`ProvenanceManifest` is written to ``data/raw/<dataset>/provenance.yaml``
and a :class:`LicensePolicy` decides whether a dataset with an unknown/missing or
disallowed license is refused outright or merely flagged.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

MANIFEST_FILENAME = "provenance.yaml"

# License strings that we treat as "no usable license recorded".
_UNKNOWN_TOKENS = {"", "unknown", "unspecified", "none", "n/a", "tbd"}


class LicenseError(RuntimeError):
    """Raised when a dataset is refused because of its license under the active policy."""


def is_license_known(license: str | None) -> bool:
    """True iff ``license`` is a concrete, non-placeholder license string."""
    if license is None:
        return False
    return license.strip().lower() not in _UNKNOWN_TOKENS


@dataclass(frozen=True)
class LicensePolicy:
    """Gate that decides whether a dataset's license is acceptable.

    Args:
        require_license: When True (default), a missing/unknown license is a hard
            refusal. When False, such a dataset is allowed but flagged.
        allowed: If set, only these license identifiers (case-insensitive) are
            accepted; anything else is treated like a refused/flagged license.
            ``None`` means "any known license is fine".
    """

    require_license: bool = True
    allowed: frozenset[str] | None = None

    def _is_allowed(self, license: str | None) -> bool:
        if not is_license_known(license):
            return False
        if self.allowed is None:
            return True
        assert license is not None
        return license.strip().lower() in {a.strip().lower() for a in self.allowed}

    def evaluate(self, license: str | None, *, dataset: str) -> tuple[bool, str | None]:
        """Apply the policy to ``license``.

        Returns ``(license_ok, warning)``. Raises :class:`LicenseError` when the
        license is unacceptable and the policy is strict (``require_license``).
        """
        if self._is_allowed(license):
            return True, None

        if not is_license_known(license):
            reason = f"dataset {dataset!r} has no usable license (got {license!r})"
        else:
            reason = (
                f"dataset {dataset!r} license {license!r} is not in the allowed set "
                f"{sorted(self.allowed) if self.allowed else None}"
            )

        if self.require_license:
            raise LicenseError(
                f"Refusing to ingest: {reason}. Set require_license=False to flag-and-keep instead."
            )
        return False, f"LICENSE FLAGGED: {reason}"


@dataclass(frozen=True)
class ProvenanceManifest:
    """The recorded origin of one dataset, persisted to ``provenance.yaml``."""

    dataset: str
    source: str  # provider name, e.g. "sklearn", "local_registry", "scraper"
    identifier: str  # dataset id within the provider
    retrieved_at: str  # ISO-8601 UTC timestamp
    checksum: str  # sha256 over the fetched data files
    license: str | None = None
    license_url: str | None = None
    license_ok: bool = True
    url: str | None = None
    n_samples: int | None = None
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "dataset": self.dataset,
            "source": self.source,
            "identifier": self.identifier,
            "url": self.url,
            "license": self.license,
            "license_url": self.license_url,
            "license_ok": self.license_ok,
            "retrieved_at": self.retrieved_at,
            "checksum": self.checksum,
            "n_samples": self.n_samples,
            "notes": self.notes,
        }
        if self.extra:
            d["extra"] = self.extra
        return d


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (provenance retrieval timestamp)."""
    return datetime.now(timezone.utc).isoformat()


def checksum_paths(paths: list[Path]) -> str:
    """Stable sha256 over the bytes of ``paths`` (sorted by name for determinism)."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: x.name):
        if not p.is_file():
            continue
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def write_manifest(dataset_dir: Path, manifest: ProvenanceManifest) -> Path:
    """Serialize ``manifest`` to ``<dataset_dir>/provenance.yaml`` and return the path."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / MANIFEST_FILENAME
    path.write_text(yaml.safe_dump(manifest.to_dict(), sort_keys=False))
    return path


def read_manifest(dataset_dir: Path) -> ProvenanceManifest | None:
    """Read back a manifest if present, else ``None``."""
    path = dataset_dir / MANIFEST_FILENAME
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text()) or {}
    return ProvenanceManifest(
        dataset=data.get("dataset", dataset_dir.name),
        source=data.get("source", "unknown"),
        identifier=data.get("identifier", ""),
        retrieved_at=data.get("retrieved_at", ""),
        checksum=data.get("checksum", ""),
        license=data.get("license"),
        license_url=data.get("license_url"),
        license_ok=bool(data.get("license_ok", True)),
        url=data.get("url"),
        n_samples=data.get("n_samples"),
        notes=data.get("notes"),
        extra=data.get("extra", {}) or {},
    )
