"""Data sourcing (M5): search/download datasets with provenance + license capture,
a crawl/scrape fallback, and de-dup / validation / auto train/val/test splitting.

The test split is always sealed via the harness :func:`metis.benchmark.seal_holdout`,
so the agent only ever receives train/val and the benchmark lockbox stays intact.
"""

from __future__ import annotations

from metis.data_sources.ingest import (
    SplitResult,
    ValidationError,
    ValidationReport,
    dedup,
    ingest_arrays,
    ingest_dataset,
    load_raw_arrays,
    split_and_seal,
    validate,
)
from metis.data_sources.provenance import (
    LicenseError,
    LicensePolicy,
    ProvenanceManifest,
    is_license_known,
    read_manifest,
    write_manifest,
)
from metis.data_sources.provider import DatasetInfo, DatasetProvider, FetchResult
from metis.data_sources.providers import (
    LocalRegistryProvider,
    ScraperProvider,
    SklearnDatasetProvider,
    build_provider_registry,
)

__all__ = [
    # provider interface
    "DatasetProvider",
    "DatasetInfo",
    "FetchResult",
    # concrete providers
    "SklearnDatasetProvider",
    "LocalRegistryProvider",
    "ScraperProvider",
    "build_provider_registry",
    # provenance + license
    "ProvenanceManifest",
    "LicensePolicy",
    "LicenseError",
    "is_license_known",
    "write_manifest",
    "read_manifest",
    # ingest
    "dedup",
    "validate",
    "split_and_seal",
    "ingest_arrays",
    "ingest_dataset",
    "load_raw_arrays",
    "SplitResult",
    "ValidationReport",
    "ValidationError",
]
