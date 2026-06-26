"""Metis CLI entrypoint.

    metis new <name>            scaffold a project directory
    metis run [name]            launch the TUI project picker + live agent feed
    metis seal <name>           seal holdout from data/processed/ into benchmark/holdout/
    metis demo <name>           run the toy PROPOSE->TRAIN->BENCHMARK pipeline (sklearn digits)
    metis benchmark <name> <variant-id>   run benchmark on a trained variant
    metis prune <name>          mark the weakest variants pruned (reversible)
    metis budget <name>         show cumulative resource usage vs. declared budgets

`run` ignores `name` for now (the TUI always shows the full picker over
PROJECTS_DIR); the agent loop itself is driven by `metis.agent`, not the CLI.
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
    p_seal.add_argument("--mode", choices=["imagenet", "numpy"], default="imagenet")
    p_seal.add_argument("--fraction", type=float, default=0.2)
    p_seal.add_argument("--seed", type=int, default=42)

    p_demo = sub.add_parser("demo", help="run the toy PROPOSE→TRAIN→BENCHMARK pipeline (digits)")
    p_demo.add_argument("name")

    p_bench = sub.add_parser("benchmark", help="run the harness benchmark on a trained variant")
    p_bench.add_argument("name")
    p_bench.add_argument("variant_id")

    p_prune = sub.add_parser("prune", help="mark the weakest variants pruned (reversible)")
    p_prune.add_argument("name")
    p_prune.add_argument("--keep-top-k", type=int, default=None)
    p_prune.add_argument("--drop-bottom-fraction", type=float, default=None)

    p_budget = sub.add_parser("budget", help="show cumulative resource usage vs. budgets")
    p_budget.add_argument("name")

    args = parser.parse_args(argv)
    if args.command == "new":
        return cmd_new(args.name)
    if args.command == "run":
        return cmd_run(args.name)
    if args.command == "seal":
        return cmd_seal(args.name, args.mode, args.fraction, args.seed)
    if args.command == "demo":
        return cmd_demo(args.name)
    if args.command == "benchmark":
        return cmd_benchmark(args.name, args.variant_id)
    if args.command == "prune":
        return cmd_prune(args.name, args.keep_top_k, args.drop_bottom_fraction)
    if args.command == "budget":
        return cmd_budget(args.name)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
