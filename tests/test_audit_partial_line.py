"""An append after a torn line must stay readable by every later process."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

from mulder.audit import AuditLog


def _torn_log(tmp_path: Path) -> Path:
    """A log whose writer died mid-way through its second line."""
    log_path = tmp_path / "audit.jsonl"
    AuditLog(log_path).log_tool_call("tc_00000001", "search", {}, "sha256:x")
    with open(log_path, "a") as fh:
        fh.write('{"type":"tool_call","tool_call_id":"tc_00000002","tool_na')
    return log_path


def test_entry_after_a_torn_line_survives_a_reload(tmp_path: Path) -> None:
    log_path = _torn_log(tmp_path)

    writer = AuditLog(log_path)
    writer.log_tool_call("tc_00000003", "search", {}, "sha256:y")
    assert writer.has_tool_call("tc_00000003")

    reloaded = AuditLog(log_path)
    assert reloaded.has_tool_call("tc_00000001")
    assert reloaded.has_tool_call("tc_00000003")
    # The torn entry itself is unrecoverable; it must not take a neighbour with it.
    assert not reloaded.has_tool_call("tc_00000002")


def test_torn_fragment_is_kept_on_its_own_line(tmp_path: Path) -> None:
    log_path = _torn_log(tmp_path)

    AuditLog(log_path).log_tool_call("tc_00000003", "search", {}, "sha256:y")

    raw = log_path.read_bytes()
    # The fragment is preserved, ended with a newline, and not glued to the
    # entry that follows it.
    assert b'"tool_na\n{"type":"tool_call","tool_call_id":"tc_00000003"' in raw


def test_intact_log_gains_no_blank_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    log = AuditLog(log_path)
    log.log_tool_call("tc_a", "search", {}, "sha256:x")
    log.log_tool_call("tc_b", "search", {}, "sha256:y")

    assert log_path.read_bytes().count(b"\n") == 2
    assert b"\n\n" not in log_path.read_bytes()


def _append_many(path: str, tag: str, count: int) -> None:
    log = AuditLog(Path(path))
    for i in range(count):
        # Large params make each append span several write() calls' worth of
        # time, so an unlocked reader of the last byte can catch it mid-write.
        log.log_tool_call(f"{tag}_{i}", "search", {"pad": "x" * 20_000}, "sha256:x")


def test_concurrent_writers_add_no_blank_lines(tmp_path: Path) -> None:
    """Appends from several processes must not look torn to one another."""
    log_path = tmp_path / "audit.jsonl"
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_append_many, args=(str(log_path), f"p{n}", 40)) for n in range(6)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
        assert proc.exitcode == 0

    lines = log_path.read_bytes().split(b"\n")
    assert lines[-1] == b""
    assert all(lines[:-1]), "blank line inserted into a healthy log"
    assert len(lines) - 1 == 240
