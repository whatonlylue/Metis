"""OS-level sandbox enforcement — the structural half of the benchmark lockbox.

The Python-level audit hook in ``pyrunner`` is advisory: a script can shed it by
spawning a fresh interpreter (``subprocess``) or by issuing raw syscalls through
``ctypes``/libc, neither of which goes through Python's audit machinery. The real
boundary lives here. ``wrap_sandboxed`` rewrites a command so the child process —
*and every process it spawns* — is denied filesystem access to the project's
``benchmark/`` subtree by the kernel itself, so the denial survives both
``subprocess`` and native ``ctypes`` calls.

Platform support:

  * **macOS (Darwin):** ``sandbox-exec`` with a generated profile that allows
    everything by default but denies ``file-read*``/``file-write*`` on a
    ``(subpath ...)`` rule covering the resolved ``benchmark/`` directory. The
    macOS sandbox is enforced at the syscall layer and is inherited by child
    processes, so a spawned ``python`` or a ``libc.open`` is denied just the same.

  * **Linux:** ``bwrap`` (bubblewrap), if present, with a ``tmpfs`` mounted over
    the ``benchmark/`` directory so the real contents are unreachable and any
    writes land in an ephemeral overlay that never touches the real tree.

  * **Anything else / tool missing:** we **fail closed** with
    ``SandboxUnavailable`` rather than run agent code without OS enforcement.
    Silently running unsandboxed would defeat the lockbox, so we refuse.
"""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path


class SandboxUnavailable(RuntimeError):
    """Raised when no OS-level sandbox is available — we refuse to run unsandboxed."""


def _real(path: str | Path) -> str:
    """Canonical absolute path with symlinks resolved (works for missing paths)."""
    return os.path.realpath(str(path))


def _scheme_quote(path: str) -> str:
    """Quote a path for a TinyScheme sandbox profile string literal."""
    escaped = path.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def macos_profile(benchmark_dir: str | Path) -> str:
    """Generate a sandbox-exec profile denying all access to *benchmark_dir*.

    ``(allow default)`` permits the interpreter to run normally; the trailing
    ``(deny ...)`` overrides it for the sealed subtree (last matching rule wins),
    covering reads (data + metadata) and writes recursively via ``subpath``.
    """
    target = _scheme_quote(_real(benchmark_dir))
    return f"(version 1)\n(allow default)\n(deny file-read* file-write* (subpath {target}))\n"


def _wrap_macos(cmd: list[str], benchmark_dir: str | Path) -> list[str]:
    exe = shutil.which("sandbox-exec")
    if exe is None:  # pragma: no cover - sandbox-exec ships with macOS
        raise SandboxUnavailable(
            "sandbox-exec not found on PATH; refusing to run agent code unsandboxed"
        )
    return [exe, "-p", macos_profile(benchmark_dir), *cmd]


def _wrap_linux(cmd: list[str], benchmark_dir: str | Path) -> list[str]:
    exe = shutil.which("bwrap")
    if exe is None:
        raise SandboxUnavailable(
            "bubblewrap (bwrap) not found; refusing to run agent code without OS-level "
            "lockbox enforcement. Install bubblewrap or run on a platform with a "
            "supported sandbox."
        )
    target = _real(benchmark_dir)
    # Bind the whole filesystem read/write, then mask the benchmark subtree with
    # an empty tmpfs: real holdout/results are unreachable and any writes land in
    # the ephemeral tmpfs rather than the real tree.
    return [
        exe,
        "--dev-bind",
        "/",
        "/",
        "--tmpfs",
        target,
        "--",
        *cmd,
    ]


def wrap_sandboxed(cmd: list[str], project_root: Path, benchmark_dir: str | Path) -> list[str]:
    """Return *cmd* wrapped so the OS denies all FS access to *benchmark_dir*.

    The returned command can be handed to ``subprocess.Popen``/``run`` exactly as
    the original would have been. The denial is kernel-enforced and inherited by
    child processes, so it cannot be shed via ``subprocess`` or ``ctypes``.

    Raises ``SandboxUnavailable`` (fail closed) when no OS sandbox is available
    for the current platform.
    """
    system = platform.system()
    if system == "Darwin":
        return _wrap_macos(cmd, benchmark_dir)
    if system == "Linux":
        return _wrap_linux(cmd, benchmark_dir)
    raise SandboxUnavailable(  # pragma: no cover - platform-dependent
        f"no OS-level sandbox available for platform {system!r}; refusing to run "
        "agent code without lockbox enforcement"
    )
