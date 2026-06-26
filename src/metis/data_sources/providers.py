"""Concrete dataset providers.

Three providers ship, all of which run **offline and deterministically** so the
test suite never touches the live internet:

* :class:`SklearnDatasetProvider` — exposes scikit-learn's bundled toy datasets
  (digits, iris, wine, breast_cancer) as "downloadable" datasets. Real arrays,
  known permissive licenses.
* :class:`LocalRegistryProvider` — a filesystem registry: each subdirectory of a
  root is a dataset (``X.npy``/``y.npy`` plus an optional ``meta.yaml`` carrying
  license/url/description). Doubles as a fixture source in tests.
* :class:`ScraperProvider` — the crawl/scrape *fallback* used when the human
  provides no data. It recursively crawls a ``file://`` / local root, treating it
  like a scraped source. Live ``http(s)`` crawling is **stubbed** behind an
  injectable ``opener`` so it is never required for tests.

Network-backed providers (Kaggle, HuggingFace, …) can be added behind the same
:class:`~metis.data_sources.provider.DatasetProvider` protocol later.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import yaml

from metis.data_sources.provenance import (
    LicensePolicy,
    ProvenanceManifest,
    checksum_paths,
    utc_now_iso,
)
from metis.data_sources.provider import DatasetInfo, FetchResult


def _matches(query: str, *fields: str) -> bool:
    """Case-insensitive substring match; an empty query matches everything."""
    q = query.strip().lower()
    if not q:
        return True
    return any(q in (f or "").lower() for f in fields)


def _build_result(
    *,
    dataset: str,
    dataset_dir: Path,
    source: str,
    identifier: str,
    data_paths: list[Path],
    license: str | None,
    license_url: str | None,
    url: str | None,
    n_samples: int | None,
    notes: str | None,
    policy: LicensePolicy | None,
) -> FetchResult:
    """Shared tail: run the license policy, write the manifest, return a result."""
    policy = policy or LicensePolicy()
    license_ok, warning = policy.evaluate(license, dataset=dataset)
    manifest = ProvenanceManifest(
        dataset=dataset,
        source=source,
        identifier=identifier,
        retrieved_at=utc_now_iso(),
        checksum=checksum_paths(data_paths),
        license=license,
        license_url=license_url,
        license_ok=license_ok,
        url=url,
        n_samples=n_samples,
        notes=notes,
    )
    from metis.data_sources.provenance import write_manifest

    write_manifest(dataset_dir, manifest)
    warnings = [warning] if warning else []
    return FetchResult(
        dataset=dataset,
        dataset_dir=dataset_dir,
        manifest=manifest,
        license_ok=license_ok,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# scikit-learn bundled toy datasets
# ---------------------------------------------------------------------------


class SklearnDatasetProvider:
    """Bundled scikit-learn datasets, surfaced as fetchable datasets."""

    name = "sklearn"

    # dataset_id -> (loader_name, human_name, license, license_url, task_type)
    _CATALOGUE: dict[str, tuple[str, str, str, str, str]] = {
        "digits": (
            "load_digits",
            "Handwritten digits (8x8)",
            "CC-BY-4.0",
            "https://archive.ics.uci.edu/dataset/80",
            "image_classification",
        ),
        "iris": (
            "load_iris",
            "Iris flower measurements",
            "CC-BY-4.0",
            "https://archive.ics.uci.edu/dataset/53",
            "tabular_classification",
        ),
        "wine": (
            "load_wine",
            "Wine cultivar chemistry",
            "CC-BY-4.0",
            "https://archive.ics.uci.edu/dataset/109",
            "tabular_classification",
        ),
        "breast_cancer": (
            "load_breast_cancer",
            "Breast cancer Wisconsin (diagnostic)",
            "CC-BY-4.0",
            "https://archive.ics.uci.edu/dataset/17",
            "tabular_classification",
        ),
    }

    def search(self, query: str, *, limit: int = 10) -> list[DatasetInfo]:
        hits: list[DatasetInfo] = []
        for ds_id, (_loader, name, lic, lic_url, task) in self._CATALOGUE.items():
            if _matches(query, ds_id, name, task):
                hits.append(
                    DatasetInfo(
                        provider=self.name,
                        dataset_id=ds_id,
                        name=name,
                        description=name,
                        license=lic,
                        license_url=lic_url,
                        url=lic_url,
                        task_type=task,
                    )
                )
            if len(hits) >= limit:
                break
        return hits

    def fetch(
        self,
        dataset_id: str,
        dest_root: Path,
        *,
        policy: LicensePolicy | None = None,
    ) -> FetchResult:
        if dataset_id not in self._CATALOGUE:
            raise KeyError(f"Unknown sklearn dataset: {dataset_id!r}")
        loader_name, name, lic, lic_url, _task = self._CATALOGUE[dataset_id]

        import numpy as np
        from sklearn import datasets as sk_datasets

        bunch = getattr(sk_datasets, loader_name)()
        X = np.asarray(bunch.data, dtype=np.float32)
        y = np.asarray(bunch.target, dtype=np.int64)

        dataset_dir = dest_root / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        np.save(dataset_dir / "X.npy", X)
        np.save(dataset_dir / "y.npy", y)

        return _build_result(
            dataset=dataset_id,
            dataset_dir=dataset_dir,
            source=self.name,
            identifier=loader_name,
            data_paths=[dataset_dir / "X.npy", dataset_dir / "y.npy"],
            license=lic,
            license_url=lic_url,
            url=lic_url,
            n_samples=int(X.shape[0]),
            notes=f"scikit-learn bundled dataset via datasets.{loader_name}()",
            policy=policy,
        )


# ---------------------------------------------------------------------------
# Local filesystem registry (also a fixture source for tests)
# ---------------------------------------------------------------------------

# Files copied verbatim when a registry/scrape dataset is fetched.
_DATA_GLOBS = ("X.npy", "y.npy", "*.csv")


class LocalRegistryProvider:
    """A registry rooted at a directory; each subdir is one dataset.

    A dataset directory may contain a ``meta.yaml`` with keys ``name``,
    ``description``, ``license``, ``license_url``, ``url`` and ``task_type``.
    Datasets with no ``license`` in ``meta.yaml`` exercise the policy's
    refuse/flag path.
    """

    name = "local_registry"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _meta(self, ds_dir: Path) -> dict[str, object]:
        meta_path = ds_dir / "meta.yaml"
        if meta_path.exists():
            return yaml.safe_load(meta_path.read_text()) or {}
        return {}

    def _dataset_dirs(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(p for p in self.root.iterdir() if p.is_dir())

    def search(self, query: str, *, limit: int = 10) -> list[DatasetInfo]:
        hits: list[DatasetInfo] = []
        for ds_dir in self._dataset_dirs():
            meta = self._meta(ds_dir)
            name = str(meta.get("name", ds_dir.name))
            desc = str(meta.get("description", ""))
            if _matches(query, ds_dir.name, name, desc):
                hits.append(
                    DatasetInfo(
                        provider=self.name,
                        dataset_id=ds_dir.name,
                        name=name,
                        description=desc,
                        license=_opt_str(meta.get("license")),
                        license_url=_opt_str(meta.get("license_url")),
                        url=_opt_str(meta.get("url")),
                        task_type=_opt_str(meta.get("task_type")),
                    )
                )
            if len(hits) >= limit:
                break
        return hits

    def fetch(
        self,
        dataset_id: str,
        dest_root: Path,
        *,
        policy: LicensePolicy | None = None,
    ) -> FetchResult:
        src_dir = self.root / dataset_id
        if not src_dir.is_dir():
            raise KeyError(f"Unknown dataset {dataset_id!r} in registry {self.root}")
        meta = self._meta(src_dir)

        dataset_dir = dest_root / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        copied = _copy_data_files(src_dir, dataset_dir)
        if not copied:
            raise FileNotFoundError(
                f"No data files ({', '.join(_DATA_GLOBS)}) found for {dataset_id!r} in {src_dir}"
            )

        return _build_result(
            dataset=dataset_id,
            dataset_dir=dataset_dir,
            source=self.name,
            identifier=str(src_dir),
            data_paths=copied,
            license=_opt_str(meta.get("license")),
            license_url=_opt_str(meta.get("license_url")),
            url=_opt_str(meta.get("url")),
            n_samples=_opt_int(meta.get("n_samples")),
            notes=str(meta.get("description", "")) or None,
            policy=policy,
        )


# ---------------------------------------------------------------------------
# Crawl/scrape fallback (offline-testable; network stubbed)
# ---------------------------------------------------------------------------


class ScraperProvider:
    """Crawl/scrape *fallback* used when the human supplies no data.

    Designed to be testable offline: point ``source_root`` at a local directory
    (or a ``file://`` URL). The provider recursively crawls it, discovering data
    files and reading a sidecar ``meta.yaml``/``LICENSE`` for provenance. Live
    ``http(s)`` crawling is stubbed: an ``opener`` callable must be injected (tests
    never do, so the network path is never exercised), otherwise fetch raises.
    """

    name = "scraper"

    def __init__(
        self,
        source_root: str | Path,
        *,
        opener: Callable[[str], bytes] | None = None,
    ) -> None:
        self.source_root = source_root
        self.opener = opener

    def _local_root(self) -> Path | None:
        root = str(self.source_root)
        if root.startswith("file://"):
            return Path(root[len("file://") :])
        if root.startswith(("http://", "https://")):
            return None
        return Path(root)

    def _require_local(self) -> Path:
        local = self._local_root()
        if local is None:
            raise NotImplementedError(
                "Live http(s) scraping is stubbed in this build. Inject an `opener` "
                "and a network crawler, or point source_root at a local/file:// path."
            )
        return local

    def search(self, query: str, *, limit: int = 10) -> list[DatasetInfo]:
        root = self._local_root()
        if root is None or not root.is_dir():
            return []
        hits: list[DatasetInfo] = []
        for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            meta = _read_meta(ds_dir)
            name = str(meta.get("name", ds_dir.name))
            if _matches(query, ds_dir.name, name, str(meta.get("description", ""))):
                hits.append(
                    DatasetInfo(
                        provider=self.name,
                        dataset_id=ds_dir.name,
                        name=name,
                        description=str(meta.get("description", "")),
                        license=_opt_str(meta.get("license")) or _read_license_file(ds_dir),
                        license_url=_opt_str(meta.get("license_url")),
                        url=f"{self.source_root}/{ds_dir.name}",
                        task_type=_opt_str(meta.get("task_type")),
                    )
                )
            if len(hits) >= limit:
                break
        return hits

    def fetch(
        self,
        dataset_id: str,
        dest_root: Path,
        *,
        policy: LicensePolicy | None = None,
    ) -> FetchResult:
        root = self._require_local()
        src_dir = root / dataset_id
        if not src_dir.is_dir():
            raise KeyError(f"Scrape source has no dataset {dataset_id!r} under {root}")
        meta = _read_meta(src_dir)

        dataset_dir = dest_root / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        copied = _copy_data_files(src_dir, dataset_dir)
        if not copied:
            raise FileNotFoundError(
                f"Scrape found no data files for {dataset_id!r} under {src_dir}"
            )

        license = _opt_str(meta.get("license")) or _read_license_file(src_dir)
        return _build_result(
            dataset=dataset_id,
            dataset_dir=dataset_dir,
            source=self.name,
            identifier=str(src_dir),
            data_paths=copied,
            license=license,
            license_url=_opt_str(meta.get("license_url")),
            url=f"{self.source_root}/{dataset_id}",
            n_samples=_opt_int(meta.get("n_samples")),
            notes="crawled from local/file:// scrape source",
            policy=policy,
        )


# ---------------------------------------------------------------------------
# Registry of providers available to the agent tools / CLI
# ---------------------------------------------------------------------------


def build_provider_registry(*, registry_root: Path | None = None) -> dict[str, object]:
    """Default set of providers. Offline by construction.

    ``registry_root`` optionally enables a :class:`LocalRegistryProvider` over a
    fixtures/registry directory.
    """
    registry: dict[str, object] = {"sklearn": SklearnDatasetProvider()}
    if registry_root is not None:
        registry["local_registry"] = LocalRegistryProvider(registry_root)
    return registry


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _opt_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _read_meta(ds_dir: Path) -> dict[str, object]:
    meta_path = ds_dir / "meta.yaml"
    if meta_path.exists():
        return yaml.safe_load(meta_path.read_text()) or {}
    return {}


def _read_license_file(ds_dir: Path) -> str | None:
    """Best-effort license discovery from a sidecar LICENSE file's first line."""
    for candidate in ("LICENSE", "LICENSE.txt", "LICENSE.md"):
        p = ds_dir / candidate
        if p.is_file():
            first = p.read_text().strip().splitlines()
            if first:
                return first[0].strip()
    return None


def _copy_data_files(src_dir: Path, dataset_dir: Path) -> list[Path]:
    """Copy known data files from ``src_dir`` into ``dataset_dir``; return the copies."""
    copied: list[Path] = []
    seen: set[Path] = set()
    for glob in _DATA_GLOBS:
        for src in sorted(src_dir.glob(glob)):
            if src in seen or not src.is_file():
                continue
            seen.add(src)
            dest = dataset_dir / src.name
            shutil.copy2(src, dest)
            copied.append(dest)
    return copied
