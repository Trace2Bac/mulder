"""A Volatility run whose plugins all failed must be reported as an error.

``run_volatility`` wrapped a failed plugin in a success response that said the
output was indexed, and ``run_volatility_batch`` returned success when every
plugin failed. Both were logged as successful calls, so they could be cited as
evidence and a background batch was marked completed. A batch in which at least
one plugin succeeded is still a success, like ``run_parallel``, and so are a
timeout whose partial output was indexed and a netscan answered by a fallback.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mulder.audit import AuditLog
from mulder.db import CaseDB
from mulder.extractors.volatility import _plugin_short_name
from mulder.server import app
from mulder.server.app import _tool_dispatch_sync
from mulder.server.tools.extract import volatility  # ensures the tools are registered
from mulder.server.tools.findings import _uncitable_refs

_VOL = "mulder.server.tools.extract.volatility"

# Stands in for the vol binary: PsList prints rows, PsScan prints rows and then
# hangs, netscan fails as on an XP image (ConnScan answers when FAKE_CONNSCAN is
# set), every other plugin fails with no output.
_FAKE_VOL = """\
import os
import sys
import time
plugin = sys.argv[-1]
if plugin.endswith(("PsList", "PsScan")) or (
    plugin.endswith("ConnScan") and os.environ.get("FAKE_CONNSCAN")
):
    print("PID\\tPPID\\tImageFileName")
    print("4\\t0\\tSystem")
    print("512\\t4\\tsmss.exe", flush=True)
    if plugin.endswith("PsScan"):
        time.sleep(60)
    sys.exit(0)
if plugin.endswith("NetScan"):
    sys.stderr.write("netscan is unsupported for this image")
else:
    sys.stderr.write("Unsatisfied requirement: kernel.layer_name")
sys.exit(1)
"""


@pytest.fixture()
def ctx(tmp_path: Path) -> Iterator[MagicMock]:
    db = CaseDB.create(case_id="vol", evidence_root=str(tmp_path), db_dir=tmp_path)
    ctx = MagicMock(db=db, audit=AuditLog(tmp_path / "vol.audit.jsonl"))
    with patch.object(app, "_ctx", ctx):
        yield ctx
    db.close()


@pytest.fixture()
def memory_path(tmp_path: Path) -> str:
    path = tmp_path / "memory.raw"
    path.write_bytes(b"\0" * 4096)
    return str(path)


@pytest.fixture()
def fake_vol(tmp_path: Path) -> Iterator[None]:
    script = tmp_path / "fake_vol.py"
    script.write_text(_FAKE_VOL)
    with patch(f"{_VOL}._find_vol_binary", return_value=[sys.executable, str(script)]):
        yield


def _audit_status(ctx: MagicMock, result: dict[str, Any]) -> object:
    entry = ctx.audit.get_tool_call(result["tool_call_id"])
    assert entry is not None
    return entry.get("status")


def _citable(ctx: MagicMock, result: dict[str, Any]) -> bool:
    return not _uncitable_refs(ctx.audit, [str(result["tool_call_id"])])


@pytest.mark.usefixtures("fake_vol")
class TestRunVolatility:
    def test_failed_plugin_is_an_error(self, ctx: MagicMock, memory_path: str) -> None:
        result = _tool_dispatch_sync["run_volatility"](plugin="cmdline", memory_path=memory_path)

        assert result["status"] == "error"
        assert result["error_type"] == "no_output"
        assert "Unsatisfied requirement" in result["error_message"]
        assert "hint" not in result
        assert _audit_status(ctx, result) == "error"
        assert not _citable(ctx, result)
        assert ctx.db.get_sources() == []

    def test_failed_netscan_keeps_its_suggestion(self, ctx: MagicMock, memory_path: str) -> None:
        result = _tool_dispatch_sync["run_volatility"](plugin="netscan", memory_path=memory_path)

        assert result["status"] == "error"
        assert result["error_type"] == "plugin_unsupported"
        assert "connscan" in str(result["suggestion"])

    def test_successful_plugin_is_unchanged(self, ctx: MagicMock, memory_path: str) -> None:
        result = _tool_dispatch_sync["run_volatility"](plugin="pslist", memory_path=memory_path)

        assert result["status"] == "success"
        assert result["source"] == "volatility.pslist"
        assert _audit_status(ctx, result) is None
        assert _citable(ctx, result)

    def test_timeout_with_partial_output_stays_a_success(
        self, ctx: MagicMock, memory_path: str
    ) -> None:
        # The partial result carries error_type "timeout", but its rows are indexed.
        with patch(f"{_VOL}.adaptive_timeout", return_value=1):
            result = _tool_dispatch_sync["run_volatility"](
                plugin="psscan", memory_path=memory_path
            )

        assert result["status"] == "success"
        assert result["source"] == "volatility.psscan"
        assert _audit_status(ctx, result) is None
        assert [s.source_name for s in ctx.db.get_sources()] == ["volatility.psscan"]

    def test_netscan_answered_by_a_fallback_stays_a_success(
        self, ctx: MagicMock, memory_path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_CONNSCAN", "1")
        result = _tool_dispatch_sync["run_volatility"](plugin="netscan", memory_path=memory_path)

        assert result["status"] == "success"
        assert _audit_status(ctx, result) is None
        assert [s.source_name for s in ctx.db.get_sources()] == ["volatility.connscan"]


def _batch(plugins: list[str], memory_path: str) -> dict[str, Any]:
    result: dict[str, Any] = _tool_dispatch_sync["run_volatility_batch"](
        plugins=plugins, memory_path=memory_path, force=True
    )
    return result


def _api_batch(
    plugins: list[str], memory_path: str, outputs: dict[str, dict[str, Any] | Exception]
) -> dict[str, Any]:
    """Run the batch through the Python API path with each plugin's outcome given."""
    known = {volatility._resolve_plugin_name(p): object() for p in outputs}

    def run_plugin(_ctx: Any, _auto: Any, _cls: Any, name: str, _path: str) -> dict[str, Any]:
        outcome = outputs[_plugin_short_name(name)]
        if isinstance(outcome, Exception):
            raise outcome
        return {"plugin": name, **outcome}

    with (
        patch(f"{_VOL}._build_volatility_context", return_value=(None, None, known)),
        patch(f"{_VOL}._run_batch_plugin", side_effect=run_plugin),
    ):
        return _batch(plugins, memory_path)


class TestRunVolatilityBatch:
    def test_every_plugin_failing_is_an_error(self, ctx: MagicMock, memory_path: str) -> None:
        result = _api_batch(
            ["pslist", "cmdline", "notaplugin"],
            memory_path,
            {
                "pslist": RuntimeError("Unsatisfied requirement"),
                "cmdline": {"status": "empty", "error_message": "Plugin produced no output"},
            },
        )

        assert result["status"] == "error"
        assert result["error_type"] == "all_plugins_failed"
        payload = result["results"]
        assert payload["plugins_succeeded"] == 0
        assert payload["plugins_failed"] == 3
        assert set(payload["per_plugin"]) == {"pslist", "cmdline", "notaplugin"}
        assert payload["per_plugin"]["notaplugin"]["error_type"] == "unknown_plugin"
        assert _audit_status(ctx, result) == "error"
        assert not _citable(ctx, result)

    def test_one_successful_plugin_keeps_the_batch_a_success(
        self, ctx: MagicMock, memory_path: str
    ) -> None:
        result = _api_batch(
            ["pslist", "cmdline"],
            memory_path,
            {
                "pslist": {"windows_indexed": 1, "line_count": 3},
                "cmdline": RuntimeError("Unsatisfied requirement"),
            },
        )

        assert result["status"] == "success"
        assert result["results"]["plugins_succeeded"] == 1
        assert result["results"]["plugins_failed"] == 1
        assert _audit_status(ctx, result) is None
        assert _citable(ctx, result)

    def test_header_only_plugins_are_not_counted_as_failures(
        self, ctx: MagicMock, memory_path: str
    ) -> None:
        result = _api_batch(
            ["pslist", "cmdline"],
            memory_path,
            {"pslist": {"status": "header_only"}, "cmdline": {"status": "header_only"}},
        )

        assert result["status"] == "success"
        assert result["results"]["plugins_header_only"] == 2
        assert _audit_status(ctx, result) is None

    @pytest.mark.usefixtures("fake_vol")
    def test_subprocess_fallback_with_every_plugin_failing_is_an_error(
        self, ctx: MagicMock, memory_path: str
    ) -> None:
        with patch(f"{_VOL}._build_volatility_context", side_effect=ImportError):
            result = _batch(["cmdline", "netscan"], memory_path)

        assert result["status"] == "error"
        assert result["error_type"] == "all_plugins_failed"
        assert result["results"]["execution_mode"] == "subprocess_fallback"
        assert _audit_status(ctx, result) == "error"

    @pytest.mark.usefixtures("fake_vol")
    def test_subprocess_fallback_with_a_successful_plugin_is_a_success(
        self, ctx: MagicMock, memory_path: str
    ) -> None:
        with patch(f"{_VOL}._build_volatility_context", side_effect=ImportError):
            result = _batch(["pslist", "cmdline"], memory_path)

        assert result["status"] == "success"
        assert result["results"]["plugins_succeeded"] == 1
        assert [s.source_name for s in ctx.db.get_sources()] == ["volatility.pslist"]
