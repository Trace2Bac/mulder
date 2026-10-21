"""BitLocker credentials must not sit on dislocker-fuse's command line.

``run_dislocker`` built ``["dislocker-fuse", "-p", <key>, ...]`` /
``["-u", <password>, ...]``.  argv is world-readable via
``/proc/<pid>/cmdline`` and is recorded by execve auditing, and because
dislocker declares ``-p``/``-u`` with an *optional* argument, a separate
token is never consumed as the value: it stayed on argv unscrubbed for the
lifetime of the FUSE daemon.

The user password now goes over stdin (bare ``-u``).  dislocker only prompts
for a recovery password on /dev/tty, so the key is passed in the attached
``--recovery-password=`` form that dislocker overwrites in argv on startup.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from mulder.server.tools.extract import misc

_RECOVERY_KEY = "111111-222222-333333-444444-555555-666666-777777-888888"
_PASSWORD = "correct horse battery staple"


def _invoke(tmp_path: Path, **creds: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run decrypt mode with subprocess mocked; return (response, run kwargs)."""
    image = tmp_path / "bitlocker.raw"
    image.write_bytes(b"\x00" * 16)
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((list(cmd), kwargs))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with (
        patch.object(misc, "require_binary", return_value=True),
        patch("mulder.server.tools.extract.misc.subprocess.run", side_effect=_fake_run),
        patch(
            "mulder.server.tools.extract.misc.tempfile.mkdtemp",
            return_value=str(tmp_path / "mnt"),
        ),
        patch.object(misc, "extract_and_index", return_value={}),
        patch.dict(misc._dislocker_mounts, clear=True),
    ):
        response = misc.run_dislocker.__wrapped__(str(image), **creds)  # type: ignore[attr-defined]

    assert len(calls) == 1
    cmd, kwargs = calls[0]
    kwargs["cmd"] = cmd
    return response, kwargs


def test_password_is_sent_on_stdin_not_argv(tmp_path: Path) -> None:
    response, kwargs = _invoke(tmp_path, password=_PASSWORD)

    cmd = kwargs["cmd"]
    assert not any(_PASSWORD in arg for arg in cmd)
    assert "-u" in cmd
    assert kwargs["input"] == _PASSWORD + "\n"
    assert _PASSWORD not in repr(response)


def test_recovery_key_is_never_a_separate_argv_token(tmp_path: Path) -> None:
    response, kwargs = _invoke(tmp_path, recovery_key=_RECOVERY_KEY)

    cmd = kwargs["cmd"]
    assert _RECOVERY_KEY not in cmd
    assert [a for a in cmd if _RECOVERY_KEY in a] == [f"--recovery-password={_RECOVERY_KEY}"]
    assert kwargs.get("input") is None
    assert _RECOVERY_KEY not in repr(response)


@pytest.mark.parametrize("creds", [{"password": _PASSWORD}, {"recovery_key": _RECOVERY_KEY}])
def test_volume_and_mount_point_follow_option_terminator(
    tmp_path: Path, creds: dict[str, str]
) -> None:
    _, kwargs = _invoke(tmp_path, **creds)

    cmd = kwargs["cmd"]
    assert cmd[-3:] == ["--", str(tmp_path / "bitlocker.raw"), str(tmp_path / "mnt")]
