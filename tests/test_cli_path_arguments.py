"""An agent-supplied path must never reach a CLI tool as an option.

Several tools append ``target_path`` / ``image_path`` / ``evidence_path`` as the
last positional argument.  A value starting with ``-`` was parsed by the tool
as a flag: ``run_clamav(target_path="--log=/some/file")`` made clamscan write
its log to an arbitrary file (and scan the server's working directory).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from mulder.server.helpers import cli_path
from mulder.server.tools.extract.misc import (
    run_clamav,
    run_exiftool,
    run_hashdeep,
    run_ssdeep,
    run_strings,
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("--log=/tmp/x", "./--log=/tmp/x"),
        ("-r", "./-r"),
        ("/evidence/disk.E01", "/evidence/disk.E01"),
        ("relative/file.bin", "relative/file.bin"),
        ("", ""),
    ],
)
def test_cli_path(path: str, expected: str) -> None:
    assert cli_path(path) == expected


@pytest.mark.parametrize("tool", [run_strings, run_exiftool, run_hashdeep, run_ssdeep])
def test_run_cli_tool_wrappers_neutralise_options(tool: Any) -> None:
    seen: list[list[str]] = []

    def _run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "out\n", "")

    with (
        patch("mulder.server.helpers.require_binary", return_value=True),
        patch("mulder.server.helpers.run_subprocess", side_effect=_run),
        patch(
            "mulder.server.extract_helpers.extract_and_index",
            return_value={"source_name": "s", "windows_indexed": 0, "line_count": 1},
        ),
    ):
        tool.__wrapped__("--help")

    assert seen and seen[0][-1] == "./--help"


@pytest.mark.skipif(shutil.which("clamscan") is None, reason="clamscan not installed")
def test_run_clamav_does_not_take_options_from_target_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive the real clamscan: ``--log=`` must not create a file."""
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    (workdir / "benign.txt").write_text("nothing to see\n")
    monkeypatch.chdir(workdir)
    log = tmp_path / "outside" / "clamscan.log"
    log.parent.mkdir()

    with patch(
        "mulder.server.tools.extract.misc.extract_and_index",
        return_value={"source_name": "clamav.scan", "windows_indexed": 0, "line_count": 0},
    ):
        run_clamav.__wrapped__(f"--log={log}")  # type: ignore[attr-defined]

    assert not log.exists()
