"""Provider-agnostic dataset-source interface.

A :class:`DatasetProvider` knows how to *search* a catalogue of datasets and to
*fetch* one into a project's ``data/raw/`` tree. The interface is deliberately
small so new sources (a Kaggle client, a HuggingFace client, a web scraper) can
be dropped in behind the same contract. Concrete offline providers live in
:mod:`metis.data_sources.providers`.

``fetch`` is the choke point that enforces the licensing guardrail: it must write
a :class:`~metis.data_sources.provenance.ProvenanceManifest` and run the dataset's
license through the active :class:`~metis.data_sources.provenance.LicensePolicy`
before the data is considered usable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from metis.data_sources.provenance import LicensePolicy, ProvenanceManifest


@dataclass(frozen=True)
class DatasetInfo:
    """A search hit: enough metadata to decide whether to fetch a dataset."""

    provider: str
    dataset_id: str
    name: str
    description: str = ""
    license: str | None = None
    license_url: str | None = None
    url: str | None = None
    n_samples: int | None = None
    task_type: str | None = None

    def summary(self) -> str:
        lic = self.license or "UNKNOWN"
        n = f", n={self.n_samples}" if self.n_samples is not None else ""
        return f"[{self.provider}] {self.dataset_id}: {self.name} (license={lic}{n})"


@dataclass(frozen=True)
class FetchResult:
    """Outcome of fetching one dataset into ``data/raw/<dataset>/``."""

    dataset: str
    dataset_dir: Path
    manifest: ProvenanceManifest
    license_ok: bool
    warnings: list[str] = field(default_factory=list)


@runtime_checkable
class DatasetProvider(Protocol):
    """The contract every dataset source implements."""

    name: str

    def search(self, query: str, *, limit: int = 10) -> list[DatasetInfo]:
        """Return up to ``limit`` datasets matching ``query`` (empty query = list all)."""
        ...

    def fetch(
        self,
        dataset_id: str,
        dest_root: Path,
        *,
        policy: LicensePolicy | None = None,
    ) -> FetchResult:
        """Materialize ``dataset_id`` under ``dest_root/<dataset>/`` with a manifest.

        ``dest_root`` is the project's ``data/raw`` directory. Implementations must
        record provenance + license and apply ``policy`` (defaulting to a strict
        :class:`LicensePolicy`) before returning.
        """
        ...
