"""Tests for the sandboxed ``run_python`` tool (M3): lockbox + budgets."""

from __future__ import annotations

import platform
import resource
import textwrap
from pathlib import Path

import pytest

from metis.projects import create_project
from metis.projects.schema import ProjectSpec, TaskType
from metis.sandbox import read_actions, run_python
from metis.sandbox.lockbox import LockboxViolation


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    spec = ProjectSpec(
        name="rp",
        description="run_python test project",
        task_type=TaskType.tabular_classification,
        target_metric="accuracy",
    )
    return create_project(tmp_path / "rp", spec)


def _write(project_root: Path, rel: str, body: str) -> str:
    path = project_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body))
    return rel


def test_runs_and_captures_output(project_root: Path) -> None:
    rel = _write(
        project_root,
        "models/v/run.py",
        """
        print("hello from sandbox")
        """,
    )
    result = run_python(project_root, rel)
    assert result.exit_code == 0
    assert not result.timed_out
    assert "hello from sandbox" in result.stdout


def test_nonzero_exit_captured(project_root: Path) -> None:
    rel = _write(
        project_root,
        "models/v/boom.py",
        """
        import sys
        sys.stderr.write("kaboom")
        sys.exit(3)
        """,
    )
    result = run_python(project_root, rel)
    assert result.exit_code == 3
    assert "kaboom" in result.stderr


def test_cwd_is_project_root(project_root: Path) -> None:
    rel = _write(
        project_root,
        "models/v/cwd.py",
        """
        import os
        print(os.getcwd())
        """,
    )
    result = run_python(project_root, rel)
    assert result.stdout.strip() == str(project_root.resolve())


def test_script_in_benchmark_is_blocked(project_root: Path) -> None:
    # Place a script inside the sealed dir (harness-side, bypassing the sandbox)
    # and confirm run_python refuses to resolve/execute it.
    evil = project_root / "benchmark" / "evil.py"
    evil.parent.mkdir(parents=True, exist_ok=True)
    evil.write_text("print('should never run')")
    with pytest.raises(LockboxViolation):
        run_python(project_root, "benchmark/evil.py")


def test_script_outside_project_is_blocked(project_root: Path) -> None:
    with pytest.raises(LockboxViolation):
        run_python(project_root, "../escape.py")


def test_cannot_read_benchmark_at_runtime(project_root: Path) -> None:
    secret = project_root / "benchmark" / "holdout" / "X.npy"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_bytes(b"top secret holdout")
    rel = _write(
        project_root,
        "models/v/peek.py",
        """
        with open("benchmark/holdout/X.npy", "rb") as f:
            print(f.read())
        """,
    )
    result = run_python(project_root, rel)
    assert result.exit_code != 0
    assert "sealed" in result.stderr.lower()


def test_cannot_write_benchmark_at_runtime(project_root: Path) -> None:
    rel = _write(
        project_root,
        "models/v/tamper.py",
        """
        with open("benchmark/results.db", "w") as f:
            f.write("cheating")
        """,
    )
    result = run_python(project_root, rel)
    assert result.exit_code != 0
    assert "sealed" in result.stderr.lower()
    assert not (project_root / "benchmark" / "results.db").exists()


def test_cannot_list_benchmark_at_runtime(project_root: Path) -> None:
    (project_root / "benchmark" / "holdout").mkdir(parents=True, exist_ok=True)
    rel = _write(
        project_root,
        "models/v/ls.py",
        """
        import os
        print(os.listdir("benchmark/holdout"))
        """,
    )
    result = run_python(project_root, rel)
    assert result.exit_code != 0
    assert "sealed" in result.stderr.lower()


_SECRET = b"TOP-SECRET-HOLDOUT-LABELS-0123456789"


def _seed_secret(project_root: Path) -> Path:
    secret = project_root / "benchmark" / "holdout" / "y.npy"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_bytes(_SECRET)
    return secret


def test_cannot_read_benchmark_via_absolute_path(project_root: Path) -> None:
    secret = _seed_secret(project_root)
    rel = _write(
        project_root,
        "models/v/abs.py",
        f"""
        data = open({str(secret)!r}, "rb").read()
        print("LEAKED:" + data.decode())
        """,
    )
    result = run_python(project_root, rel)
    assert result.exit_code != 0
    assert _SECRET.decode() not in result.stdout


def test_cannot_read_benchmark_via_os_open(project_root: Path) -> None:
    secret = _seed_secret(project_root)
    rel = _write(
        project_root,
        "models/v/osopen.py",
        f"""
        import os
        fd = os.open({str(secret)!r}, os.O_RDONLY)
        print("LEAKED:" + os.read(fd, 4096).decode())
        """,
    )
    result = run_python(project_root, rel)
    assert result.exit_code != 0
    assert _SECRET.decode() not in result.stdout


def test_cannot_read_benchmark_via_ctypes_libc(project_root: Path) -> None:
    # ctypes/libc.open bypasses Python's audit machinery entirely; only an
    # OS-level (kernel) sandbox can deny it.
    secret = _seed_secret(project_root)
    rel = _write(
        project_root,
        "models/v/ct.py",
        f"""
        import ctypes, ctypes.util, os, sys
        name = ctypes.util.find_library("c") or "libc.dylib"
        libc = ctypes.CDLL(name, use_errno=True)
        fd = libc.open({str(secret)!r}.encode(), 0)
        if fd < 0:
            print("BLOCKED errno", ctypes.get_errno())
            sys.exit(7)
        buf = os.read(fd, 4096)
        print("LEAKED:" + buf.decode())
        """,
    )
    result = run_python(project_root, rel)
    # Either the open is denied (fd < 0) or the whole process is sandboxed; in
    # no case should the secret bytes appear.
    assert _SECRET.decode() not in result.stdout


def test_cannot_read_benchmark_via_spawned_subprocess(project_root: Path) -> None:
    # Escaping the audit hook by spawning a fresh interpreter: the child inherits
    # the OS sandbox, so the relative/absolute open is still denied.
    secret = _seed_secret(project_root)
    rel = _write(
        project_root,
        "models/v/spawn.py",
        f"""
        import subprocess, sys
        child = "print(open({str(secret)!r}, 'rb').read().decode())"
        r = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        print("CHILD_RC", r.returncode)
        """,
    )
    result = run_python(project_root, rel)
    assert _SECRET.decode() not in result.stdout
    assert "CHILD_RC 0" not in result.stdout  # child failed to read


def test_cannot_write_benchmark_via_spawned_subprocess(project_root: Path) -> None:
    # A child interpreter must not be able to tamper with results.db either.
    (project_root / "benchmark").mkdir(parents=True, exist_ok=True)
    rel = _write(
        project_root,
        "models/v/spawnwrite.py",
        """
        import subprocess, sys
        child = "open('benchmark/results.db','w').write('cheat')"
        subprocess.run([sys.executable, "-c", child])
        """,
    )
    run_python(project_root, rel)
    assert not (project_root / "benchmark" / "results.db").exists()


def test_timeout_kills_long_script(project_root: Path) -> None:
    rel = _write(
        project_root,
        "models/v/slow.py",
        """
        import time
        time.sleep(30)
        print("done")
        """,
    )
    result = run_python(project_root, rel, timeout_s=1.0)
    assert result.timed_out
    assert result.exit_code != 0
    assert "done" not in result.stdout
    assert result.duration_s < 10  # killed promptly, not after 30s


def test_memory_cap_is_configured(project_root: Path) -> None:
    cap_mb = 2048
    rel = _write(
        project_root,
        "models/v/mem.py",
        """
        import resource
        soft, _ = resource.getrlimit(resource.RLIMIT_AS)
        print(soft)
        """,
    )
    result = run_python(project_root, rel, memory_mb=cap_mb)
    assert result.exit_code == 0, result.stderr
    soft = int(result.stdout.strip())

    if platform.system() == "Linux":
        # Linux honours RLIMIT_AS; the cap must be applied to the child.
        assert soft == cap_mb * 1024 * 1024
    elif platform.system() == "Darwin":
        # macOS does NOT enforce RLIMIT_AS — setrlimit is a no-op for address
        # space. The wall-clock timeout is the enforced memory backstop here
        # (see pyrunner._apply_limits). We make this explicit rather than
        # silently passing as if memory were capped.
        pytest.skip("macOS does not enforce RLIMIT_AS; wall-clock timeout is the backstop")
    else:  # pragma: no cover - other platforms
        if soft == resource.RLIM_INFINITY:
            pytest.skip("platform does not enforce RLIMIT_AS")
        assert soft == cap_mb * 1024 * 1024


def test_invocation_is_logged(project_root: Path) -> None:
    rel = _write(project_root, "models/v/ok.py", "print('ok')")
    run_python(project_root, rel)
    actions = read_actions(project_root)
    assert any(a["tool"] == "run_python" and a["ok"] for a in actions)
