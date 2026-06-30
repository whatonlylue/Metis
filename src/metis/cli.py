"""Metis CLI entrypoint.

    metis new <name>            scaffold a project directory
    metis run [name]            launch the TUI: project chat-boxes, leaderboard, agent chat
    metis seal <name>           seal holdout from data/processed/ into benchmark/holdout/
    metis fetch <name> --dataset <id>   download a dataset into data/raw with provenance+license
    metis ingest <name> --dataset <id>  dedup/validate/split a dataset and seal the test holdout
    metis demo <name>           run the toy PROPOSE->TRAIN->BENCHMARK pipeline (sklearn digits)
    metis benchmark <name> <variant-id>   run benchmark on a trained variant
    metis prune <name>          mark the weakest variants pruned (reversible)
    metis budget <name>         show cumulative resource usage vs. declared budgets
    metis robustness <name> <variant-id>  score a variant under holdout corruptions
    metis export <name> <variant-id>      export a variant (ONNX or pickle bundle)
    metis bundle <name> <variant-id>      build a reproducibility bundle

`run` ignores `name` for now (the TUI always shows every project under
PROJECTS_DIR as a chat-box in the left rail); pick one to chat with its agent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from metis.projects import create_project
from metis.projects.schema import ProjectSpec, TaskType
from metis.tui import run_tui

PROJECTS_DIR = Path("projects")


def cmd_new(name: str) -> int:
    root = PROJECTS_DIR / name
    spec = ProjectSpec(name=name, description="TODO", task_type=TaskType.image_classification)
    try:
        create_project(root, spec)
    except FileExistsError:
        print(f"Project already exists: {root}", file=sys.stderr)
        return 1
    print(f"Created project at {root}")
    print("Next: fill in project.yaml, then `metis run`.")
    return 0


def cmd_run(_name: str | None) -> int:
    run_tui(PROJECTS_DIR)
    return 0


def cmd_seal(name: str, mode: str, fraction: float, seed: int) -> int:
    from metis.benchmark import seal_holdout

    root = PROJECTS_DIR / name
    if not root.exists():
        print(f"Project not found: {root}", file=sys.stderr)
        return 1
    try:
        holdout = seal_holdout(root, mode=mode, fraction=fraction, seed=seed)  # type: ignore[arg-type]
    except Exception as exc:
        print(f"seal failed: {exc}", file=sys.stderr)
        return 1
    print(f"Holdout sealed at {holdout}")
    return 0


def cmd_demo(name: str) -> int:
    """Run the toy PROPOSE -> TRAIN -> BENCHMARK pipeline on sklearn digits."""
    from metis.benchmark import get_leaderboard
    from metis.projects import load_project
    from metis.projects.schema import ProjectSpec, TaskType
    from metis.training import run_toy_pipeline

    root = PROJECTS_DIR / name
    if not root.exists():
        spec = ProjectSpec(
            name=name,
            description="classify 8x8 handwritten digits (toy demo)",
            task_type=TaskType.image_classification,
            classes=[str(d) for d in range(10)],
            target_metric="accuracy",
        )
        create_project(root, spec)
        print(f"Created project at {root}")

    try:
        records = run_toy_pipeline(root)
    except Exception as exc:
        print(f"demo failed: {exc}", file=sys.stderr)
        return 1

    metric = load_project(root).target_metric
    rows = get_leaderboard(root / "benchmark", task_metric_name=metric, n=25)
    print(f"\nLeaderboard for {name!r} ({len(records)} variants):")
    header = f"{'Rank':>4}  {'Variant':<16}  {metric:>10}  {'Params':>10}  {'Size MB':>8}  {'p50 ms':>8}  {'samp/s':>10}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rows, 1):
        print(
            f"{i:>4}  {str(r['variant_id']):<16}  {r['task_metric_value']:>10.4f}  "
            f"{r['param_count']:>10,}  {r['model_size_mb']:>8.3f}  "
            f"{r['latency_ms_p50']:>8.3f}  {r['throughput_sps']:>10,.0f}"
        )
    return 0


def cmd_prune(name: str, keep_top_k: int | None, drop_bottom_fraction: float | None) -> int:
    from metis.benchmark import prune_project

    root = PROJECTS_DIR / name
    if not root.exists():
        print(f"Project not found: {root}", file=sys.stderr)
        return 1
    pruned = prune_project(
        root, keep_top_k=keep_top_k, drop_bottom_fraction=drop_bottom_fraction, reason="cli prune"
    )
    if not pruned:
        print("Nothing pruned (no policy configured, or only the top variant remains).")
    else:
        print(f"Pruned {len(pruned)} variant(s): {', '.join(pruned)}")
    return 0


def cmd_budget(name: str) -> int:
    from metis.benchmark import compute_budget_status

    root = PROJECTS_DIR / name
    if not root.exists():
        print(f"Project not found: {root}", file=sys.stderr)
        return 1
    s = compute_budget_status(root)

    def rem(v: object) -> str:
        return "unlimited" if v is None else str(v)

    print(
        f"wall-clock: {s.wall_clock_minutes_used:.2f} min used (remaining {rem(s.wall_clock_minutes_remaining)})"
    )
    print(f"variants trained: {s.variants_trained} (remaining {rem(s.variants_remaining)})")
    print(f"dollars: ${s.dollars_used:.2f} used (remaining {rem(s.dollars_remaining)})")
    print(f"STOP: {s.should_stop}")
    for reason in s.reasons:
        print(f"  - {reason}")
    return 0


def cmd_fetch(name: str, provider: str, dataset: str, registry_root: str | None) -> int:
    """Fetch a dataset into data/raw/<dataset>/ with provenance + license capture."""
    from metis.data_sources import build_provider_registry
    from metis.projects import load_project, record_data_source
    from metis.projects.schema import DataSourceRef

    root = PROJECTS_DIR / name
    if not root.exists():
        print(f"Project not found: {root}", file=sys.stderr)
        return 1
    try:
        spec = load_project(root)
        from metis.data_sources import LicensePolicy

        pol = spec.data.license_policy
        policy = LicensePolicy(
            require_license=pol.require_license,
            allowed=frozenset(pol.allowed_licenses) if pol.allowed_licenses else None,
        )
    except Exception:
        from metis.data_sources import LicensePolicy

        policy = LicensePolicy()

    registry = build_provider_registry(registry_root=Path(registry_root) if registry_root else None)
    prov = registry.get(provider)
    if prov is None:
        print(f"Unknown provider {provider!r}. Available: {sorted(registry)}", file=sys.stderr)
        return 1
    try:
        result = prov.fetch(dataset, root / "data" / "raw", policy=policy)
    except Exception as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1
    m = result.manifest
    record_data_source(
        root,
        DataSourceRef(
            dataset=m.dataset,
            source=m.source,
            identifier=m.identifier,
            url=m.url,
            license=m.license,
            license_ok=m.license_ok,
            checksum=m.checksum,
            retrieved_at=m.retrieved_at,
        ),
    )
    print(f"Fetched {m.dataset!r} -> {result.dataset_dir}")
    print(f"  license={m.license or 'UNKNOWN'} (ok={result.license_ok})  n_samples={m.n_samples}")
    for w in result.warnings:
        print(f"  {w}")
    return 0


def cmd_ingest(name: str, dataset: str) -> int:
    """De-dup, validate, split a fetched dataset and seal the test holdout."""
    from metis.data_sources import ingest_dataset
    from metis.projects import load_project

    root = PROJECTS_DIR / name
    if not root.exists():
        print(f"Project not found: {root}", file=sys.stderr)
        return 1
    spec = load_project(root)
    split = spec.data.split
    try:
        result = ingest_dataset(
            root,
            root / "data" / "raw" / dataset,
            train=split.train,
            val=split.val,
            test=split.test,
            seed=spec.data.split_seed,
            classes=spec.classes,
        )
    except Exception as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Ingested {dataset!r}: train={result.train_size}, val={result.val_size}, "
        f"test sealed={result.test_size} (in benchmark/holdout, agent-invisible)"
    )
    print(result.report.summary())
    return 0


def cmd_robustness(name: str, variant_id: str, seed: int | None) -> int:
    """Run the harness-side robustness benchmark on a trained variant."""
    from metis.benchmark import BenchmarkRunner

    root = PROJECTS_DIR / name
    if not root.exists():
        print(f"Project not found: {root}", file=sys.stderr)
        return 1
    record = BenchmarkRunner().run_robustness(root, variant_id, seed=seed)
    if record.error and record.clean_score is None:
        print(f"No robustness score: {record.error}", file=sys.stderr)
        return 1
    print(f"clean {record.task_metric_name}={record.clean_score:.4f}")
    for cname, score in record.per_corruption.items():
        print(f"  {cname:<18} {score:.4f}")
    if record.aggregate_robustness is not None:
        print(f"aggregate robustness (retention) = {record.aggregate_robustness:.3f}")
    return 0


def cmd_export(name: str, variant_id: str, no_onnx: bool) -> int:
    """Export a trained variant to a portable artifact (ONNX or pickle bundle)."""
    from metis.portability import export_variant

    root = PROJECTS_DIR / name
    if not root.exists():
        print(f"Project not found: {root}", file=sys.stderr)
        return 1
    try:
        result = export_variant(root, variant_id, prefer_onnx=not no_onnx)
    except Exception as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 1
    print(f"Exported {variant_id!r} as {result.format} -> {result.path}")
    print(f"  sha256={result.checksum}")
    if result.format != "onnx" and not result.onnx_available:
        print("  (ONNX deps not installed — wrote a self-contained pickle bundle instead)")
    return 0


def cmd_bundle(name: str, variant_id: str) -> int:
    """Assemble a reproducibility bundle (recipe + code + env + data manifest + results)."""
    from metis.portability import build_repro_bundle

    root = PROJECTS_DIR / name
    if not root.exists():
        print(f"Project not found: {root}", file=sys.stderr)
        return 1
    try:
        result = build_repro_bundle(root, variant_id)
    except Exception as exc:
        print(f"bundle failed: {exc}", file=sys.stderr)
        return 1
    print(f"Bundle for {variant_id!r}:")
    print(f"  directory: {result.directory}")
    print(f"  archive:   {result.archive}")
    print(f"  contents:  {', '.join(result.manifest['contents'])}")
    return 0


def cmd_benchmark(name: str, variant_id: str) -> int:
    from metis.benchmark import BenchmarkRunner

    root = PROJECTS_DIR / name
    if not root.exists():
        print(f"Project not found: {root}", file=sys.stderr)
        return 1
    runner = BenchmarkRunner()
    record = runner.run(root, variant_id)
    if record.task_metric_value is not None:
        print(
            f"{record.task_metric_name}={record.task_metric_value:.4f}  "
            f"size={record.model_size_mb} MB  params={record.param_count}"
        )
    else:
        print(f"No score: {record.error}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="metis")
    sub = parser.add_subparsers(dest="command", required=True)

    p_new = sub.add_parser("new", help="scaffold a new project")
    p_new.add_argument("name")

    p_run = sub.add_parser("run", help="launch the TUI + agent loop")
    p_run.add_argument("name", nargs="?")

    p_seal = sub.add_parser("seal", help="seal holdout from data/processed/ into benchmark/")
    p_seal.add_argument("name")
    p_seal.add_argument("--mode", choices=["auto", "imagenet", "numpy"], default="auto")
    p_seal.add_argument("--fraction", type=float, default=0.2)
    p_seal.add_argument("--seed", type=int, default=42)

    p_demo = sub.add_parser("demo", help="run the toy PROPOSE→TRAIN→BENCHMARK pipeline (digits)")
    p_demo.add_argument("name")

    p_fetch = sub.add_parser("fetch", help="download a dataset into data/raw with provenance")
    p_fetch.add_argument("name")
    p_fetch.add_argument("--provider", default="sklearn")
    p_fetch.add_argument("--dataset", required=True)
    p_fetch.add_argument("--registry-root", default=None, help="local registry root (optional)")

    p_ingest = sub.add_parser("ingest", help="dedup/validate/split a dataset and seal the holdout")
    p_ingest.add_argument("name")
    p_ingest.add_argument("--dataset", required=True)

    p_bench = sub.add_parser("benchmark", help="run the harness benchmark on a trained variant")
    p_bench.add_argument("name")
    p_bench.add_argument("variant_id")

    p_prune = sub.add_parser("prune", help="mark the weakest variants pruned (reversible)")
    p_prune.add_argument("name")
    p_prune.add_argument("--keep-top-k", type=int, default=None)
    p_prune.add_argument("--drop-bottom-fraction", type=float, default=None)

    p_budget = sub.add_parser("budget", help="show cumulative resource usage vs. budgets")
    p_budget.add_argument("name")

    p_robust = sub.add_parser(
        "robustness", help="run the harness robustness benchmark on a variant"
    )
    p_robust.add_argument("name")
    p_robust.add_argument("variant_id")
    p_robust.add_argument("--seed", type=int, default=None)

    p_export = sub.add_parser("export", help="export a trained variant (ONNX or pickle bundle)")
    p_export.add_argument("name")
    p_export.add_argument("variant_id")
    p_export.add_argument("--no-onnx", action="store_true", help="force the pickle-bundle fallback")

    p_bundle = sub.add_parser("bundle", help="build a reproducibility bundle for a variant")
    p_bundle.add_argument("name")
    p_bundle.add_argument("variant_id")

    args = parser.parse_args(argv)
    if args.command == "new":
        return cmd_new(args.name)
    if args.command == "run":
        return cmd_run(args.name)
    if args.command == "seal":
        return cmd_seal(args.name, args.mode, args.fraction, args.seed)
    if args.command == "demo":
        return cmd_demo(args.name)
    if args.command == "fetch":
        return cmd_fetch(args.name, args.provider, args.dataset, args.registry_root)
    if args.command == "ingest":
        return cmd_ingest(args.name, args.dataset)
    if args.command == "benchmark":
        return cmd_benchmark(args.name, args.variant_id)
    if args.command == "prune":
        return cmd_prune(args.name, args.keep_top_k, args.drop_bottom_fraction)
    if args.command == "budget":
        return cmd_budget(args.name)
    if args.command == "robustness":
        return cmd_robustness(args.name, args.variant_id, args.seed)
    if args.command == "export":
        return cmd_export(args.name, args.variant_id, args.no_onnx)
    if args.command == "bundle":
        return cmd_bundle(args.name, args.variant_id)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
