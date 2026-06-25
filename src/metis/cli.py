"""Metis CLI entrypoint.

    metis new <name>     scaffold a project directory
    metis run <name>     launch the TUI + agent loop (M1+)

This is an M0 stub: `new` works; `run` is wired up in later milestones.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from metis.projects import create_project
from metis.projects.schema import ProjectSpec, TaskType

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


def cmd_run(name: str) -> int:
    print(f"`metis run {name}` is not implemented yet (see docs/ROADMAP.md, M1).")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="metis")
    sub = parser.add_subparsers(dest="command", required=True)

    p_new = sub.add_parser("new", help="scaffold a new project")
    p_new.add_argument("name")

    p_run = sub.add_parser("run", help="launch the TUI + agent loop")
    p_run.add_argument("name")

    args = parser.parse_args(argv)
    if args.command == "new":
        return cmd_new(args.name)
    if args.command == "run":
        return cmd_run(args.name)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
