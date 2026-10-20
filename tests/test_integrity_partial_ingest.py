"""A source whose indexing was cut short must not verify as intact.

``insert_windows`` used to rewrite ``windows_hash`` after every batch with the
hash of the windows stored so far, so a run that died between batches left a
hash that matched its partial windows and verified. The hash is now recorded
once, after the last batch, and a source still marked pending is incomplete.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from unittest.mock import patch

import pytest

from mulder.audit import AuditLog
from mulder.db import CaseDB
from mulder.index.correlator import Correlator
from mulder.server.app import ServerContext
from mulder.server.extract_helpers import extract_and_index
from mulder.server.tools import case

# One window per line, three windows per batch.
_OUTPUT = "".join(f"row {i}\n" for i in range(7))


@pytest.fixture
def ctx(tmp_case_db: CaseDB, tmp_audit_log: AuditLog) -> Iterator[ServerContext]:
    server_ctx = ServerContext(
        case_id="test-case",
        db=tmp_case_db,
        correlator=Correlator(tmp_case_db),
        audit=tmp_audit_log,
    )
    with (
        patch("mulder.server.app._ctx", server_ctx),
        patch("mulder.server.extract_helpers._INSERT_BATCH_SIZE", 3),
        patch("mulder.server.extract_helpers._WINDOW_CHAR_BUDGET", 6),
    ):
        yield server_ctx


def _index_failing_on_batch(ctx: ServerContext, failing_batch: int) -> None:
    real_insert = ctx.db.insert_windows
    calls = 0

    def insert(source_id: int, windows: list[object]) -> None:
        nonlocal calls
        calls += 1
        if calls == failing_batch:
            raise OSError("disk full")
        real_insert(source_id, windows)  # type: ignore[arg-type]

    with (
        patch.object(ctx.db, "insert_windows", side_effect=insert),
        pytest.raises(OSError, match="disk full"),
    ):
        extract_and_index(_OUTPUT, "partial.output", "/evidence/p", "test")


def _verify() -> dict[str, object]:
    return case.verify_evidence_integrity.__wrapped__()  # type: ignore[attr-defined, no-any-return]


def _status_of(result: dict[str, object], source_name: str) -> object:
    sources = result["sources"]
    assert isinstance(sources, list)
    return next(r["status"] for r in sources if r["source_name"] == source_name)


def test_complete_source_hash_covers_every_batch(ctx: ServerContext) -> None:
    extract_and_index(_OUTPUT, "full.output", "/evidence/f", "test")

    windows = ctx.db.get_windows_by_source("full.output")
    assert len(windows) == 7
    expected = hashlib.blake2b(digest_size=32)
    for w in windows:
        expected.update(w.raw_text.encode())
    src = next(s for s in ctx.db.get_sources() if s.source_name == "full.output")
    assert src.windows_hash == "blake2b:" + expected.hexdigest()

    result = _verify()
    assert _status_of(result, "full.output") == "verified"
    assert result["status"] == "all_verified"


@pytest.mark.parametrize("failing_batch", [1, 2])
def test_interrupted_indexing_is_incomplete(ctx: ServerContext, failing_batch: int) -> None:
    extract_and_index(_OUTPUT, "full.output", "/evidence/f", "test")
    _index_failing_on_batch(ctx, failing_batch)

    result = _verify()
    assert _status_of(result, "partial.output") == "incomplete"
    assert _status_of(result, "full.output") == "verified"
    assert result["incomplete_count"] == 1
    assert result["status"] == "integrity_issues_detected"


def test_source_without_hash_is_still_reported_as_legacy(ctx: ServerContext) -> None:
    ctx.db.register_source("legacy.output", "/evidence/l", "blake2b:x", "test", 1)

    result = _verify()
    assert _status_of(result, "legacy.output") == "no_hash_recorded"
    assert result["status"] == "all_verified"
