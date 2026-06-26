"""Sandboxed ``run_python`` — execute an agent-written script under budgets.

This is the tool the agent uses to run a training script it wrote. The script
runs in an isolated subprocess that:

  * is confined to the active project directory (cwd = project_root, and the
    script path is resolved through the lockbox so it can never live in or
    point at ``benchmark/``);
  * is wrapped in an **OS-level sandbox** (``ossandbox.wrap_sandboxed``) that
    makes the sealed ``benchmark/`` subtree unreachable for filesystem read and
    write at the *kernel* layer — the load-bearing boundary;
  * has its wall-clock time capped (``timeout_s``) — on expiry the whole
    process group is killed;
  * has its address space capped (``memory_mb``) via ``RLIMIT_AS`` where the
    platform supports it (a no-op on macOS — see ``_apply_limits``);
  * additionally carries a ``sys.addaudithook`` that denies opens into
    ``benchmark/`` as *defense-in-depth*.

The lockbox boundary is structural, not advisory. ``ossandbox.wrap_sandboxed``
is the guarantee: because it is enforced by the kernel, it survives a script
that spawns a fresh interpreter via ``subprocess`` (the child inherits the
sandbox) or issues raw syscalls via ``ctypes``/libc (which never reach Python's
audit machinery). The audit hook below is kept as a secondary in-interpreter
check, but it is NOT relied upon as the boundary. ``resolve_within_project``
separately stops the agent from *placing* or *naming* a script inside
``benchmark/``.

Every invocation is appended to ``runs/actions.jsonl`` via ``log_action``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from metis.sandbox.lockbox import SEALED_DIRNAME, resolve_within_project
from metis.sandbox.ossandbox import wrap_sandboxed
from metis.sandbox.runlog import log_action

# Harness-authored bootstrap. Defense-in-depth only: it installs an audit hook
# that denies any file open whose resolved path is inside the sealed benchmark
# directory, then runs the agent's script as __main__. The real boundary is the
# OS sandbox applied to the whole command (see wrap_sandboxed); this hook is a
# secondary check that cannot, on its own, contain a subprocess or ctypes call.
# The sealed dir is passed via the environment and read once at startup.
_BOOTSTRAP = r"""
import os, runpy, sys
from pathlib import Path

_sealed = Path(os.environ.pop("METIS_SEALED_DIR")).resolve()
_script = sys.argv[1]
sys.argv = [_script, *sys.argv[2:]]


def _under_sealed(raw):
    if not isinstance(raw, (str, bytes, os.PathLike)):
        return False
    try:
        p = Path(os.fsdecode(raw))
        if not p.is_absolute():
            p = Path.cwd() / p
        p = p.resolve()
    except Exception:
        return False
    return p == _sealed or _sealed in p.parents


def _audit(event, args):
    # File access funnels through "open"; directory listings through
    # "os.scandir"/"os.listdir". Block any that resolve into benchmark/.
    if event == "open" and args and _under_sealed(args[0]):
        raise PermissionError("benchmark/ is sealed; run_python cannot access it")
    if event in ("os.scandir", "os.listdir") and args and _under_sealed(args[0]):
        raise PermissionError("benchmark/ is sealed; run_python cannot access it")


sys.addaudithook(_audit)
runpy.run_path(_script, run_name="__main__")
"""


@dataclass
class RunResult:
    """Outcome of a sandboxed script execution."""

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float


def _apply_limits(memory_mb: int | None):  # pragma: no cover - runs in child
    """Return a preexec_fn that starts a new session and caps memory.

    ``RLIMIT_AS`` is honoured on Linux but is effectively a no-op on macOS
    (Darwin), where the kernel does not enforce an address-space cap. On macOS
    the wall-clock ``timeout_s`` is the enforced memory backstop: a runaway
    allocation that thrashes is killed when the timeout expires. We therefore
    treat the setrlimit call as best-effort and never claim hard enforcement on
    platforms that ignore it.
    """

    def _preexec() -> None:
        os.setsid()
        if memory_mb is not None:
            import resource

            cap = memory_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
            except (ValueError, OSError):
                # Some platforms refuse RLIMIT_AS; the timeout remains as a
                # backstop. Best-effort by design.
                pass

    return _preexec


def run_python(
    project_root: Path,
    script_path: str | Path,
    *,
    timeout_s: float = 120.0,
    memory_mb: int | None = None,
    args: list[str] | None = None,
) -> RunResult:
    """Execute *script_path* (inside the project) under time/memory budgets.

    The script is run with ``cwd`` set to the project root and is blocked from
    touching ``benchmark/`` both at resolution time (lockbox) and at runtime
    (audit hook). Captures stdout/stderr/exit code and logs the call.
    """
    log_args = {
        "script": str(script_path),
        "timeout_s": timeout_s,
        "memory_mb": memory_mb,
        "args": args or [],
    }
    try:
        target = resolve_within_project(project_root, script_path)
        if not target.is_file():
            raise FileNotFoundError(f"script not found: {script_path!r}")
    except Exception as exc:
        log_action(project_root, "run_python", log_args, ok=False, error=str(exc))
        raise

    sealed_dir = (project_root.resolve() / SEALED_DIRNAME).resolve()
    env = {
        **os.environ,
        "METIS_SEALED_DIR": str(sealed_dir),
        # Keep child output unbuffered so we still capture it on a kill.
        "PYTHONUNBUFFERED": "1",
    }

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_BOOTSTRAP)
        bootstrap_path = Path(f.name)

    cmd = [sys.executable, str(bootstrap_path), str(target), *(args or [])]
    # Wrap the whole command in the OS-level sandbox so benchmark/ is unreachable
    # at the kernel layer for this process and any child it spawns. This is the
    # real lockbox boundary; the in-interpreter audit hook is only secondary.
    cmd = wrap_sandboxed(cmd, project_root.resolve(), sealed_dir)
    started = time.perf_counter()
    timed_out = False
    proc = subprocess.Popen(
        cmd,
        cwd=str(project_root.resolve()),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=_apply_limits(memory_mb),
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(proc)
        stdout, stderr = proc.communicate()
        stderr = (stderr or "") + f"\n[metis] killed: exceeded {timeout_s}s wall-clock budget"
        exit_code = proc.returncode
    finally:
        bootstrap_path.unlink(missing_ok=True)

    duration = time.perf_counter() - started
    result = RunResult(
        exit_code=exit_code,
        stdout=stdout or "",
        stderr=stderr or "",
        timed_out=timed_out,
        duration_s=duration,
    )
    log_action(
        project_root,
        "run_python",
        log_args,
        ok=(not timed_out and exit_code == 0),
        error=(result.stderr.strip()[:500] or None) if result.exit_code != 0 else None,
    )
    return result


def _kill_group(proc: subprocess.Popen[str]) -> None:
    """SIGKILL the child's process group, falling back to the process itself."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass
