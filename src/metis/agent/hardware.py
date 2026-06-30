"""Detect the host's compute hardware so the agent can make sane training choices.

The agent was previously blind to the machine it runs on: it would, for example,
refuse to train on an Apple-Silicon "CPU" believing it too slow, never realising a
powerful integrated GPU was sitting right there reachable through torch's MPS
backend. :func:`describe_hardware` produces a short human-readable block that
``session`` appends to the system prompt so the agent knows the chip, core/RAM
budget, and — critically — which accelerator (CUDA / MPS) it should target.

Detection is best-effort and must never raise: a probe failure degrades to
"unknown" rather than breaking session start-up.
"""

from __future__ import annotations

import os
import platform
import subprocess


def _sysctl(key: str) -> str | None:
    """Read a macOS sysctl value, or None if unavailable/non-macOS."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    val = out.stdout.strip()
    return val or None


def _bytes_to_gib(n: int) -> str:
    return f"{n / (1024 ** 3):.0f} GB"


def _total_ram_bytes() -> int | None:
    # macOS / BSD expose hw.memsize; Linux exposes it via sysconf pages.
    mem = _sysctl("hw.memsize")
    if mem and mem.isdigit():
        return int(mem)
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def _chip_name() -> str | None:
    """Human-readable CPU/SoC name (e.g. 'Apple M2 Max')."""
    # Apple Silicon and Intel Macs both report a friendly string here.
    brand = _sysctl("machdep.cpu.brand_string")
    if brand:
        return brand
    # Linux fallback: first model-name line from /proc/cpuinfo.
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _accelerator() -> str:
    """Describe the best torch-visible accelerator and how to use it."""
    try:
        import torch
    except ImportError:
        return (
            "torch is NOT installed in the base runtime, so only numpy + "
            "scikit-learn (CPU) are available. Use torch image templates only "
            "if a torch template is offered."
        )

    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001 - probe must not raise
            name = "CUDA GPU"
        return (
            f"A CUDA GPU is available ({name}). Train on it with "
            "device='cuda' and move model + tensors to the GPU."
        )

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return (
            "An Apple-Silicon GPU is available through torch's MPS (Metal) "
            "backend. Use device='mps' (NOT 'cpu') for torch training — moving "
            "the model and input tensors to MPS gives a large speed-up over the "
            "CPU on this machine. Do not dismiss this hardware as 'too slow'; "
            "the integrated GPU is well-suited to small CNN training. Note: a few "
            "ops lack MPS kernels — if one errors, set the env var "
            "PYTORCH_ENABLE_MPS_FALLBACK=1 rather than abandoning the GPU."
        )

    return (
        "No GPU accelerator (CUDA or MPS) is visible to torch; training runs on "
        "CPU. Prefer compact architectures and modest epoch counts."
    )


def describe_hardware() -> str:
    """Return a compact, prompt-ready description of the host compute hardware."""
    chip = _chip_name() or "unknown CPU"
    cores = os.cpu_count() or "?"
    ram_bytes = _total_ram_bytes()
    ram = _bytes_to_gib(ram_bytes) if ram_bytes else "unknown"
    arch = platform.machine()
    system = platform.system()

    lines = [
        "HOST HARDWARE (use this to choose architectures, batch sizes, and the "
        "training device — do NOT assume the machine is slow):",
        f"• Chip: {chip} ({arch}, {system})",
        f"• Logical CPU cores: {cores}",
        f"• System RAM: {ram}",
        f"• Accelerator: {_accelerator()}",
    ]
    return "\n".join(lines)
