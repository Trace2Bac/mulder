"""``search(source=[...])`` searches any of the listed sources (#242).

``source`` was typed ``str`` and concatenated with ``".%"`` in the DB
layer, so the list a model passes through ``run_parallel`` blew up as a
``TypeError`` traceback instead of a tool error.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import anyio
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from mulder.db import CaseDB
from mulder.models import WindowRow
from mulder.server import app
from mulder.server.jobs import JobStore
from mulder.server.tools import core


def _add(db: CaseDB, name: str, lines: list[str]) -> int:
    sid = db.register_source(
        source_name=name, source_path="/evidence/x", source_hash="h", extractor="t", line_count=1
    )
    db.insert_windows(
        sid,
        [
            WindowRow(
                window_id=None,
                source_id=sid,
                line_start=i,
                line_end=i,
                event_time=None,
                raw_text=t,
            )
            for i, t in enumerate(lines, 1)
        ],
    )
    return sid


@pytest.fixture
def ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[CaseDB]:
    db = CaseDB.create(case_id="c242", evidence_root="/evidence", db_dir=tmp_path)
    _add(db, "tsk.filelist", ["secret plan.docx", "nothing here"])
    _add(db, "registry.system", ["USBSTOR secret serial"])
    _add(db, "evtx.security", ["secret logon 4624"])

    monkeypatch.setattr(app, "_ctx", SimpleNamespace(case_id="c242", db=db, audit=Mock()))
    yield db
    db.close()


def _search(**kwargs: Any) -> dict[str, Any]:
    resp: dict[str, Any] = core.search.__wrapped__(query="secret", **kwargs)  # type: ignore[attr-defined]
    return resp


def _sources(resp: dict[str, Any]) -> set[str]:
    return {r["source_name"] for r in resp["results"]}


@pytest.mark.parametrize("regex", [False, True])
def test_source_list_searches_any_of_them(ctx: CaseDB, regex: bool) -> None:
    resp = _search(source=["tsk.filelist", "registry.system"], regex=regex)

    assert resp["status"] == "success"
    assert resp["total_matches"] == 2
    assert _sources(resp) == {"tsk.filelist", "registry.system"}
    assert resp["source"] == ["tsk.filelist", "registry.system"]


@pytest.mark.parametrize("regex", [False, True])
def test_source_string_still_scopes(ctx: CaseDB, regex: bool) -> None:
    resp = _search(source="registry", regex=regex)

    assert resp["status"] == "success"
    assert _sources(resp) == {"registry.system"}


def test_source_list_with_evidence_path(ctx: CaseDB) -> None:
    resp = _search(source=["tsk.filelist", "evtx.security"], evidence_path="x")

    assert _sources(resp) == {"tsk.filelist", "evtx.security"}


@pytest.fixture(params=["mcp", "parallel", "background"])
def dispatch_search(
    ctx: CaseDB, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[..., dict[str, Any]]]:
    """Exercise the public validation boundaries, including real background workers."""
    monkeypatch.setattr(app, "async_wait_for_resources", AsyncMock())
    monkeypatch.setattr(app, "wait_for_resources", lambda *args: None)
    monkeypatch.setattr(app, "_get_tool_limiter", lambda: None)
    store = JobStore(max_workers=1, tool_dispatch=app._tool_dispatch_sync)
    monkeypatch.setattr(app, "_job_store", store)

    async def call(args: dict[str, Any]) -> dict[str, Any]:
        if request.param == "mcp":
            try:
                response = await app.mcp.call_tool("search", args)
            except ToolError as exc:
                return {"error": str(exc)}
            data = response.model_dump()
            result: dict[str, Any] = data.get("structuredContent") or json.loads(
                data["content"][0]["text"]
            )
            return result
        tasks = [{"tool": "search", "args": args}]
        if request.param == "parallel":
            response_dict = await app.run_parallel(tasks)
            return response_dict["parallel_results"][0]["result"]  # type: ignore[no-any-return]
        submitted = app._tool_dispatch_sync["start_extraction_batch"](tasks)
        if submitted.get("tasks_rejected"):
            return {"error": submitted["tasks_rejected"][0]["error"]}
        assert store.wait_for_batch(submitted["batch_id"], timeout=5)
        completed = store.get_completed_results(submitted["batch_id"])
        assert completed and len(completed) == 1
        return completed[0]["result"]  # type: ignore[no-any-return]

    def invoke(**kwargs: Any) -> dict[str, Any]:
        return anyio.run(call, {"query": "secret", **kwargs})

    yield invoke
    store.shutdown(wait=True)


@pytest.mark.parametrize("regex", [False, True])
@pytest.mark.parametrize("evidence_path, expected", [(None, 3), ("missing", 0)])
def test_empty_source_list_keeps_evidence_filter(
    dispatch_search: Callable[..., dict[str, Any]],
    regex: bool,
    evidence_path: str | None,
    expected: int,
) -> None:
    resp = dispatch_search(source=[], regex=regex, evidence_path=evidence_path)
    assert resp["status"] == "success"
    assert resp["total_matches"] == expected


@pytest.mark.parametrize(
    "args",
    [{"source": 42}, {"source": ["tsk", 7]}, {"queries": "secret"}, {"exclude_sources": "tsk"}],
)
def test_wrong_types_rejected_consistently(
    dispatch_search: Callable[..., dict[str, Any]], args: dict[str, Any]
) -> None:
    resp = dispatch_search(**args)
    assert "error" in resp
    assert next(iter(args)) in resp["error"]


def test_numeric_arguments_coerced_consistently(
    dispatch_search: Callable[..., dict[str, Any]],
) -> None:
    resp = dispatch_search(max_results="1", regex=True)
    assert resp["status"] == "success"
    assert resp["total_matches"] == 3
    assert len(resp["results"]) == 1
