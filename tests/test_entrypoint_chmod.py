"""The container entrypoint must not make the cases directory world-writable.

``scripts/entrypoint.sh`` runs as root and the documented ``docker run``
bind-mounts a host directory at ``/home/mulder/.mulder/cases``. It used to run
``chmod -R a+rwX`` on that path, so every case database, report and
``*.audit.jsonl`` chain-of-custody log on the host became readable and
writable by any local account. The preceding ``chown -R mulder:mulder``
already gives the container user ownership, so owner bits are all it needs.

The script is run with ``chown``/``chmod``/``mkdir``/``gosu`` stubbed on
``PATH`` so the mode it hands ``chmod`` can be recorded and then replayed with
the real ``chmod`` against a scratch tree.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

_ENTRYPOINT = Path(__file__).resolve().parent.parent / "scripts" / "entrypoint.sh"
_CASES = "/home/mulder/.mulder/cases"
_BASH = shutil.which("bash")
_CHMOD = shutil.which("chmod")

pytestmark = pytest.mark.skipif(
    _BASH is None or _CHMOD is None, reason="bash and chmod are required"
)


def _recorded_cases_chmod(tmp_path: Path) -> list[list[str]]:
    """Run the entrypoint with logging stubs; return chmod argv for cases."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.log"
    for name in ("chown", "chmod", "mkdir", "cp", "gosu"):
        stub = bindir / name
        stub.write_text(f'#!/bin/sh\necho "{name} $*" >> "{log}"\n')
        stub.chmod(0o755)
    env = {"PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}"}
    assert _BASH is not None
    subprocess.run([_BASH, str(_ENTRYPOINT), "true"], env=env, check=True)
    calls = [line.split() for line in log.read_text().splitlines()]
    assert ["gosu", "mulder", "true"] in calls
    return [c[1:] for c in calls if c[0] == "chmod" and c[-1] == _CASES]


def test_entrypoint_never_grants_group_or_other_access(tmp_path: Path) -> None:
    chmods = _recorded_cases_chmod(tmp_path)
    assert chmods, "entrypoint no longer touches the cases directory mode"
    for argv in chmods:
        mode = argv[-2]
        for clause in mode.split(","):
            who, _, perms = clause.partition("+")
            assert not (perms and (who in ("", "a") or set(who) & {"g", "o"})), (
                f"entrypoint widens cases permissions with chmod {mode}"
            )


def test_entrypoint_chmod_leaves_cases_tree_owner_only_writable(
    tmp_path: Path,
) -> None:
    """Replaying the recorded chmod must strip world-write left by old runs."""
    chmods = _recorded_cases_chmod(tmp_path)
    cases = tmp_path / "cases"
    case_dir = cases / "case-1"
    case_dir.mkdir(parents=True)
    audit = case_dir / "case-1.audit.jsonl"
    audit.write_text("{}\n")
    for p in (cases, case_dir, audit):
        p.chmod(0o777 if p.is_dir() else 0o666)

    assert _CHMOD is not None
    for argv in chmods:
        subprocess.run([_CHMOD, *argv[:-1], str(cases)], check=True)

    for p in (cases, case_dir, audit):
        mode = p.stat().st_mode
        assert not mode & (stat.S_IWGRP | stat.S_IWOTH), f"{p} is {oct(mode)}"
        assert mode & stat.S_IRUSR and mode & stat.S_IWUSR
    assert cases.stat().st_mode & stat.S_IXUSR
