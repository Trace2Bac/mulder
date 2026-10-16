"""Tests for the tshark fields requested by run_pcap_analysis modes.

tshark refuses the whole command when any ``-e`` field is unknown, so one
stale field name makes a mode fail on every capture.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from mulder.server.tools.extract import pcap

_MODE_FUNCS: dict[str, Callable[[str, int], str]] = {
    "summary": pcap._pcap_summary,
    "conversations": pcap._pcap_conversations,
    "dns": pcap._pcap_dns,
    "http": pcap._pcap_http,
    "smtp": pcap._pcap_smtp,
    "tls": pcap._pcap_tls,
    "beaconing": pcap._pcap_beaconing,
    "tunneling": pcap._pcap_tunneling,
}

# ``text`` is a tshark pseudo-field that is not listed by ``tshark -G fields``.
_PSEUDO_FIELDS = {"text"}

needs_tshark = pytest.mark.skipif(shutil.which("tshark") is None, reason="tshark not installed")


def _requested_fields(mode: str) -> set[str]:
    """Return every ``-e`` field the mode passes to tshark."""
    calls: list[list[str]] = []

    def fake_run_tshark(
        args: list[str], *_a: object, **_kw: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    with patch.object(pcap, "_run_tshark", side_effect=fake_run_tshark):
        _MODE_FUNCS[mode]("/nonexistent.pcap", 100)
    return {args[i + 1] for args in calls for i, a in enumerate(args) if a == "-e"}


def _known_fields() -> set[str]:
    out = subprocess.run(["tshark", "-G", "fields"], capture_output=True, text=True, check=True)
    return {cols[2] for line in out.stdout.splitlines() if len(cols := line.split("\t")) > 2}


def _write_pcap(path: Path) -> None:
    """Write one DNS A response and two SMTP lines (a request and a reply)."""

    def ip4(addr: str) -> bytes:
        return bytes(int(x) for x in addr.split("."))

    def frame(src: str, dst: str, proto: int, l4: bytes) -> bytes:
        ip = struct.pack(
            "!BBHHHBBH4s4s", 0x45, 0, 20 + len(l4), 1, 0, 64, proto, 0, ip4(src), ip4(dst)
        )
        return b"\x00" * 6 + b"\x00" * 6 + b"\x08\x00" + ip + l4

    question = b"\x07example\x03com\x00" + struct.pack("!HH", 1, 1)
    answer = b"\xc0\x0c" + struct.pack("!HHIH", 1, 1, 300, 4) + ip4("93.184.216.34")
    dns = struct.pack("!HHHHHH", 0x1234, 0x8180, 1, 1, 0, 0) + question + answer
    packets = [
        frame("8.8.8.8", "10.0.0.5", 17, struct.pack("!HHHH", 53, 40000, 8 + len(dns), 0) + dns)
    ]

    seq = {25: 5000, 50000: 1000}
    for sport, dport, line in (
        (50000, 25, b"MAIL FROM:<jean@m57.biz>\r\n"),
        (25, 50000, b"250 2.1.0 Sender OK\r\n"),
    ):
        tcp = struct.pack(
            "!HHIIBBHHH", sport, dport, seq[sport], seq[dport], 5 << 4, 0x18, 65535, 0, 0
        )
        src, dst = ("10.0.0.5", "10.0.0.25") if sport == 50000 else ("10.0.0.25", "10.0.0.5")
        packets.append(frame(src, dst, 6, tcp + line))
        seq[sport] += len(line)

    with path.open("wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for i, p in enumerate(packets):
            f.write(struct.pack("<IIII", 1700000000 + i, 0, len(p), len(p)) + p)


class TestRequestedFields:
    """The field names each mode asks tshark for."""

    def test_dns_mode_requests_dns_a(self) -> None:
        """dns.resp.addr only existed in Wireshark 1.4-1.10; dns.a replaced it."""
        fields = _requested_fields("dns")
        assert "dns.a" in fields
        assert "dns.resp.addr" not in fields

    def test_smtp_mode_requests_smtp_rsp_parameter(self) -> None:
        """The reply parameter field is smtp.rsp.parameter."""
        fields = _requested_fields("smtp")
        assert "smtp.rsp.parameter" in fields
        assert "smtp.response.parameter" not in fields

    @needs_tshark
    @pytest.mark.parametrize("mode", sorted(_MODE_FUNCS))
    def test_fields_known_to_installed_tshark(self, mode: str) -> None:
        """Every requested field exists in the installed tshark."""
        unknown = _requested_fields(mode) - _known_fields() - _PSEUDO_FIELDS
        assert not unknown


@needs_tshark
class TestModesOnCapture:
    """dns and smtp modes run against a real capture."""

    def test_dns_mode_returns_answer_address(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.pcap"
        _write_pcap(path)
        out = pcap._pcap_dns(str(path), 100)
        assert "tshark error" not in out
        assert "example.com" in out
        assert "93.184.216.34" in out

    def test_smtp_mode_returns_request_and_reply(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.pcap"
        _write_pcap(path)
        out = pcap._pcap_smtp(str(path), 100)
        assert "tshark error" not in out
        assert "FROM:<jean@m57.biz>" in out
        assert "2.1.0 Sender OK" in out
