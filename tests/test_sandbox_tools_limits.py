"""Tests for the context-window caps on read_file / list_dir (issue: 1.6M-token burn).

Large reads and directory listings must not flood the agent's context: read_file
windows by line + char, and list_dir output is rendered through format_listing,
which summarizes big directories by extension. Lockbox resolution must stay intact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from metis.sandbox import LockboxViolation, format_listing, list_dir, read_file
from metis.sandbox.tools import LIST_MAX_ENTRIES, READ_MAX_CHARS, READ_MAX_LINES


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    (tmp_path / "benchmark").mkdir()
    (tmp_path / "data").mkdir()
    return tmp_path


def test_read_file_returns_small_file_unchanged(project_root: Path) -> None:
    (project_root / "data" / "small.txt").write_text("arch: cnn\n")
    assert read_file(project_root, "data/small.txt") == "arch: cnn\n"


def test_read_file_windows_large_file_and_footers(project_root: Path) -> None:
    body = "".join(f"line {i}\n" for i in range(1000))
    (project_root / "data" / "big.txt").write_text(body)
    out = read_file(project_root, "data/big.txt")
    assert out.count("\n") <= READ_MAX_LINES + 2  # window + footer
    assert "of 1000" in out
    assert "line 0\n" in out and "line 399\n" in out
    assert "line 400\n" not in out


def test_read_file_offset_window(project_root: Path) -> None:
    body = "".join(f"line {i}\n" for i in range(1000))
    (project_root / "data" / "big.txt").write_text(body)
    out = read_file(project_root, "data/big.txt", offset=500, limit=10)
    assert "line 500\n" in out and "line 509\n" in out
    assert "line 499\n" not in out
    assert "showing lines 501-510 of 1000" in out


def test_read_file_char_cap(project_root: Path) -> None:
    body = "x" * (READ_MAX_CHARS + 5000) + "\n"
    (project_root / "data" / "wide.txt").write_text(body)
    out = read_file(project_root, "data/wide.txt")
    assert "char-capped" in out
    # Body (minus footer) never exceeds the hard char cap.
    assert len(out) <= READ_MAX_CHARS + 64


def test_list_dir_small_dir_plain_list() -> None:
    assert format_listing(["a.txt", "b.txt"]) == "a.txt\nb.txt"


def test_format_listing_summarizes_large_dir() -> None:
    entries = sorted(f"img_{i:04d}.png" for i in range(1000)) + ["labels.csv"]
    out = format_listing(entries)
    assert "1001 entries" in out
    assert ".png: 1000" in out
    assert ".csv: 1" in out
    assert f"showing {LIST_MAX_ENTRIES} of 1001 entries" in out


def test_format_listing_summary_flag_forces_breakdown() -> None:
    out = format_listing(["a.png", "b.png", "c.jpg"], summary=True)
    assert "by extension" in out
    assert ".png: 2" in out


def test_caps_preserve_lockbox(project_root: Path) -> None:
    with pytest.raises(LockboxViolation):
        read_file(project_root, "benchmark/results.db")
    with pytest.raises(LockboxViolation):
        list_dir(project_root, "benchmark/holdout")
