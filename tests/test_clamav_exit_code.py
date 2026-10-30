"""Regression tests for a clamscan run that failed being reported as a clean scan."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from mulder.server.tools.extract.misc import run_clamav

# Recorded from clamscan 1.5.4 with an empty signature database directory.
NO_DATABASE = (
    "LibClamAV Error: cli_loaddbdir: No supported database files found in /var/lib/clamav\n"
    "ERROR: Can't open file or directory\n"
)


@pytest.fixture
def evidence(tmp_path: Path) -> Path:
    path = tmp_path / "evidence"
    path.mkdir()
    (path / "a.txt").write_text("hello\n")
    return path


def _invoke(target: Path, proc: subprocess.CompletedProcess[str]) -> tuple[Any, list[str]]:
    indexed: list[str] = []

    def _record(raw: str, *args: object, **kwargs: object) -> dict[str, object]:
        indexed.append(raw)
        return {"source_name": "clamav.scan", "windows_indexed": 0, "line_count": 0}

    with (
        patch("mulder.server.tools.extract.misc.require_binary", return_value=True),
        patch("mulder.server.tools.extract.misc.subprocess.run", return_value=proc),
        patch("mulder.server.tools.extract.misc.extract_and_index", side_effect=_record),
    ):
        result = run_clamav.__wrapped__(str(target))  # type: ignore[attr-defined]
    return result, indexed


def test_a_failed_scan_is_an_error_not_zero_detections(evidence: Path) -> None:
    proc = subprocess.CompletedProcess(
        args=["clamscan"], returncode=2, stdout="", stderr=NO_DATABASE
    )

    result, indexed = _invoke(evidence, proc)

    assert result["status"] == "error"
    assert result["error_type"] == "tool_failed"
    message = str(result["error_message"])
    assert "clamscan exited 2" in message
    assert "No supported database files found" in message
    assert indexed == []


def test_a_detection_is_still_a_success(evidence: Path) -> None:
    proc = subprocess.CompletedProcess(
        args=["clamscan"],
        returncode=1,
        stdout=f"{evidence}/a.txt: Win.Test.EICAR_HDB-1 FOUND\n",
        stderr="",
    )

    result, indexed = _invoke(evidence, proc)

    assert result["status"] == "success"
    assert indexed == [f"{evidence}/a.txt: Win.Test.EICAR_HDB-1 FOUND"]


def test_a_clean_scan_is_still_a_success(evidence: Path) -> None:
    proc = subprocess.CompletedProcess(
        args=["clamscan"], returncode=0, stdout=f"{evidence}/a.txt: OK\n", stderr=""
    )

    result, indexed = _invoke(evidence, proc)

    assert result["status"] == "success"
    assert indexed == [f"{evidence}/a.txt: OK"]


def test_output_produced_before_a_non_zero_exit_is_kept(evidence: Path) -> None:
    proc = subprocess.CompletedProcess(
        args=["clamscan"],
        returncode=2,
        stdout=f"{evidence}/a.txt: OK\n",
        stderr=f"{evidence}/b.txt: Can't open file or directory ERROR\n",
    )

    result, indexed = _invoke(evidence, proc)

    assert result["status"] == "success"
    assert indexed == [f"{evidence}/a.txt: OK"]
