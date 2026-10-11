"""run_parallel may only dispatch tools the calling session could call itself."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import anyio
import pytest

from mulder.server import app
from mulder.server.tool_access import ALLOWED_TOOLS_ENV, Role, get_tools_for_role, may_dispatch

_calls: list[str] = []


async def _allowed_tool() -> dict[str, str]:
    _calls.append("allowed_tool")
    return {"status": "ok"}


async def _forbidden_tool() -> dict[str, str]:
    _calls.append("forbidden_tool")
    return {"status": "ok"}


def _run(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _calls.clear()
    stubs = {"allowed_tool": _allowed_tool, "forbidden_tool": _forbidden_tool}
    with (
        patch.dict(app._tool_dispatch, stubs),
        patch("mulder.server.app.has_ctx", return_value=False),
    ):
        result = anyio.run(app.run_parallel, tasks)
    return [r["result"] for r in result["parallel_results"]]


def test_unset_env_restricts_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A standalone ``mulder serve`` has no session allowlist."""
    monkeypatch.delenv(ALLOWED_TOOLS_ENV, raising=False)

    assert may_dispatch("anything")
    results = _run([{"tool": "forbidden_tool"}])

    assert results[0]["status"] == "ok"


def test_task_outside_the_sessions_allowlist_is_refused(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(ALLOWED_TOOLS_ENV, "mcp__mulder__allowed_tool,mcp__mulder__wait")

    with caplog.at_level("WARNING"):
        results = _run([{"tool": "forbidden_tool"}, {"tool": "allowed_tool"}])

    assert "not available through run_parallel" in results[0]["error"]
    assert results[1]["status"] == "ok"
    assert _calls == ["allowed_tool"]
    assert "forbidden_tool" in caplog.text


def test_a_tool_another_role_may_call_is_still_refused_for_this_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check is the caller's own allowlist, not 'some executor may call it'."""
    # update_finding is not an executor tool, but an analyst session may call it.
    executor = get_tools_for_role(Role.EXTRACT_EXECUTOR)
    monkeypatch.setenv(ALLOWED_TOOLS_ENV, ",".join(executor))

    assert not may_dispatch("update_finding")
    assert not may_dispatch("deduplicate_findings")
    assert may_dispatch("run_volatility")


@pytest.mark.parametrize("tool", ["run_parallel", "start_extraction_batch"])
def test_dispatchers_are_held_to_the_same_rule(monkeypatch: pytest.MonkeyPatch, tool: str) -> None:
    monkeypatch.setenv(ALLOWED_TOOLS_ENV, "mcp__mulder__allowed_tool")

    assert not may_dispatch(tool)
    [result] = _run([{"tool": tool, "args": {"tasks": [{"tool": "forbidden_tool"}]}}])

    assert "not available through run_parallel" in result["error"]
    assert _calls == []


def test_empty_allowlist_refuses_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOWED_TOOLS_ENV, "")

    assert not may_dispatch("allowed_tool")
