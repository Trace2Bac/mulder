"""Regression tests for find_suspicious_processes' anomaly_score.

``_score_and_sort_results`` read ``item["reasons"]``, but the only caller,
``find_suspicious_processes``, stores them under ``suspicion_reasons``.  Every
process scored 0 and the "most anomalous first" sort left the list in PID
order.  Four keys in ``_SEVERITY_WEIGHTS`` also did not match the reason
names ``_analyze_pid`` emits, so those reasons fell back to the default
weight even with the key fixed.
"""

from __future__ import annotations

from mulder.models import WindowRow
from mulder.server.tools.composite.core import _SEVERITY_WEIGHTS, _score_and_sort_results
from mulder.server.tools.composite.process import _analyze_pid


def _win(text: str) -> WindowRow:
    return WindowRow(source_id=1, line_start=1, line_end=1, event_time=None, raw_text=text)


def _analyze(pid: int, **kwargs: object) -> list[str]:
    defaults: dict[str, object] = {
        "pid_names": {pid: "evil.exe"},
        "parent_map": {},
        "malfind_pids": {},
        "cmdline_pids": {},
        "netscan_pids": {},
        "pstree_pids": {},
    }
    defaults.update(kwargs)
    entry = _analyze_pid(pid, **defaults)  # type: ignore[arg-type]
    assert entry is not None
    reasons: list[str] = entry["suspicion_reasons"]
    return reasons


def test_score_uses_suspicion_reasons() -> None:
    results = [{"pid": 3000, "suspicion_reasons": ["malfind_injection"]}]
    _score_and_sort_results(results)
    assert results[0]["anomaly_score"] == _SEVERITY_WEIGHTS["malfind_injection"]


def test_most_anomalous_process_sorts_first() -> None:
    results = [
        {"pid": 1200, "suspicion_reasons": ["dangerous_privilege"]},
        {"pid": 3000, "suspicion_reasons": ["malfind_injection"]},
        {"pid": 4100, "suspicion_reasons": ["hidden_process", "malfind_injection"]},
    ]
    _score_and_sort_results(results)
    assert [r["pid"] for r in results] == [4100, 3000, 1200]


def test_every_weight_key_is_a_reason_analyze_pid_emits() -> None:
    pid = 4100
    emitted: set[str] = set()
    emitted |= set(_analyze(pid, malfind_pids={pid: [_win(f"{pid}\t0x400000")]}))
    emitted |= set(_analyze(pid, pslist_pids={}, psscan_pids={pid: [_win(f"{pid}\t600")]}))
    emitted |= set(
        _analyze(pid, netscan_pids={pid: [_win(f"10.0.0.5:49711 203.0.113.9:4444\t{pid}")]})
    )
    emitted |= set(_analyze(pid, cmdline_pids={pid: [_win(f"{pid}\tpowershell -enc AAAA")]}))
    emitted |= set(
        _analyze(
            pid,
            pid_names={pid: "cmd.exe", 600: "services.exe"},
            parent_map={pid: 600},
        )
    )
    emitted |= set(_analyze(pid, dlllist_pids={pid: [_win(f"{pid}\tC:\\Users\\Public\\x.dll")]}))
    emitted |= set(_analyze(pid, envars_pids={pid: [_win(f"{pid}\tComSpec\tC:\\x.exe")]}))
    assert set(_SEVERITY_WEIGHTS) <= emitted, set(_SEVERITY_WEIGHTS) - emitted
