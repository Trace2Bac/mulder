"""The LiteLLM proxy is keyed per run, loopback-only, and verified before use."""

from __future__ import annotations

import io
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from mulder.orchestrator.proxy import (
    ProxyManager,
    _build_proxy_config,
    _free_port,
    _listener_owned_by,
    _wait_for_health,
    fetch_model_windows,
)

MODEL = "bedrock/test"


def _start(pm: ProxyManager, port: int = 4000) -> MagicMock:
    """Run ``pm.start()`` with litellm mocked; return the ``Popen`` mock."""
    with (
        patch("shutil.which", return_value="/usr/local/bin/litellm"),
        patch("mulder.orchestrator.proxy.litellm_reasoning", return_value={}),
        patch("mulder.orchestrator.proxy._free_port", return_value=port),
        patch("mulder.orchestrator.proxy._wait_for_health", return_value=True),
        patch("mulder.orchestrator.proxy.fetch_model_windows", return_value={}),
        patch("subprocess.Popen") as popen,
    ):
        pm.start()
    return popen


def _bearer(req: urllib.request.Request) -> str:
    return str(req.get_header("Authorization", "")).removeprefix("Bearer ")


def _answer(status_for_key: dict[str, int]) -> MagicMock:
    """A fake ``urlopen``: liveliness answers 200, ``/model_group/info``
    answers per bearer token (401 for any token not listed)."""

    def urlopen(req: urllib.request.Request, timeout: float = 0) -> MagicMock:
        url = req.full_url
        status = 200
        if url.endswith("/model_group/info"):
            status = status_for_key.get(_bearer(req), 401)
        if status >= 400:
            raise urllib.error.HTTPError(url, status, "x", {}, None)  # type: ignore[arg-type]
        resp = MagicMock()
        resp.__enter__.return_value.status = status
        return resp

    return MagicMock(side_effect=urlopen)


def _alive() -> MagicMock:
    proc = MagicMock()
    proc.poll.return_value = None
    return proc


class TestMasterKey:
    def test_key_is_random_per_run(self) -> None:
        a, b = ProxyManager(models=[MODEL]), ProxyManager(models=[MODEL])
        assert a.env_overrides["ANTHROPIC_AUTH_TOKEN"] != b.env_overrides["ANTHROPIC_AUTH_TOKEN"]
        assert len(a.env_overrides["ANTHROPIC_AUTH_TOKEN"]) >= 40

    def test_generated_config_references_env_not_value(self) -> None:
        config = _build_proxy_config([MODEL], 4000)
        assert config["general_settings"]["master_key"] == "os.environ/LITELLM_MASTER_KEY"

    def test_key_reaches_proxy_via_env_only(self) -> None:
        pm = ProxyManager(models=[MODEL])
        key = pm.env_overrides["ANTHROPIC_AUTH_TOKEN"]
        written: list[str] = []
        with patch("pathlib.Path.write_text", lambda _s, text, **_k: written.append(text)):
            popen = _start(pm)
        cmd, env = popen.call_args.args[0], popen.call_args.kwargs["env"]
        assert env["LITELLM_MASTER_KEY"] == key
        assert all(key not in arg for arg in cmd)
        assert key not in written[0]
        assert yaml.safe_load(written[0])["general_settings"]["master_key"].startswith(
            "os.environ/"
        )

    def test_fetch_model_windows_sends_given_key(self) -> None:
        resp = MagicMock()
        resp.__enter__.return_value = io.StringIO('{"data": []}')
        with patch("urllib.request.urlopen", return_value=resp) as urlopen:
            fetch_model_windows(4000, "k-1")
        assert _bearer(urlopen.call_args.args[0]) == "k-1"


class TestLoopback:
    def test_proxy_binds_loopback(self) -> None:
        cmd = _start(ProxyManager(models=[MODEL])).call_args.args[0]
        assert cmd[cmd.index("--host") + 1] == "127.0.0.1"

    def test_clients_use_loopback(self) -> None:
        pm = ProxyManager(models=[MODEL], port=4000)
        assert pm.env_overrides["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4000"


class TestPortSelection:
    def test_free_preferred_port_is_kept(self) -> None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        assert _free_port(port) == port

    def test_taken_port_is_replaced(self) -> None:
        with socket.socket() as squatter:
            squatter.bind(("127.0.0.1", 0))
            squatter.listen()
            taken = squatter.getsockname()[1]
            chosen = _free_port(taken)
        assert chosen != taken
        assert chosen > 0

    def test_manager_uses_chosen_port(self) -> None:
        pm = ProxyManager(models=[MODEL], port=4000)
        cmd = _start(pm, port=45678).call_args.args[0]
        assert cmd[cmd.index("--port") + 1] == "45678"
        assert pm.port == 45678
        assert pm.env_overrides["ANTHROPIC_BASE_URL"].endswith(":45678")


def _fake_proc(tmp_path: Path, port: int, holder: int, parents: dict[int, int]) -> Path:
    """A minimal ``/proc``: one LISTEN socket on *port* held by *holder*."""
    proc = tmp_path / "proc"
    (proc / "net").mkdir(parents=True)
    header = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt"
    header += "   uid  timeout inode\n"
    row = f"   0: 0100007F:{port:04X} 00000000:0000 0A 00000000:00000000 00:00000000"
    row += " 00000000  1000        0 424242 1\n"
    (proc / "net" / "tcp").write_text(header + row)
    (proc / "net" / "tcp6").write_text(header)
    for pid, ppid in parents.items():
        (proc / str(pid) / "fd").mkdir(parents=True)
        (proc / str(pid) / "stat").write_text(f"{pid} (litellm) S {ppid} 1 1 0\n")
    (proc / str(holder) / "fd" / "7").symlink_to("socket:[424242]")
    return proc


class TestListenerOwnership:
    def test_listener_of_child_is_ours(self, tmp_path: Path) -> None:
        proc = _fake_proc(tmp_path, 4000, holder=200, parents={200: 100, 100: 1})
        assert _listener_owned_by(4000, 200, proc) is True

    def test_listener_of_descendant_is_ours(self, tmp_path: Path) -> None:
        proc = _fake_proc(tmp_path, 4000, holder=300, parents={300: 200, 200: 100})
        assert _listener_owned_by(4000, 200, proc) is True

    def test_listener_of_other_pid_is_not_ours(self, tmp_path: Path) -> None:
        proc = _fake_proc(tmp_path, 4000, holder=999, parents={999: 1, 200: 100})
        assert _listener_owned_by(4000, 200, proc) is False

    def test_unseen_holder_is_not_ours(self, tmp_path: Path) -> None:
        proc = _fake_proc(tmp_path, 4000, holder=999, parents={999: 1, 200: 100})
        (proc / "999" / "fd" / "7").unlink()
        assert _listener_owned_by(4000, 200, proc) is False

    def test_without_proc_it_cannot_tell(self, tmp_path: Path) -> None:
        assert _listener_owned_by(4000, 200, tmp_path / "missing") is None

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc")
    def test_real_listener(self) -> None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        code = (
            "import socket, time; s = socket.socket(); "
            "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
            f"s.bind(('127.0.0.1', {port})); s.listen(); print('up', flush=True); time.sleep(30)"
        )
        child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)
        try:
            assert child.stdout is not None
            assert child.stdout.readline().strip() == b"up"
            assert _listener_owned_by(port, child.pid) is True
            assert _listener_owned_by(port, child.pid + 1_000_000) is False
        finally:
            child.kill()
            child.wait()


class TestWaitForHealth:
    @pytest.fixture(autouse=True)
    def owned(self) -> Iterator[MagicMock]:
        with patch("mulder.orchestrator.proxy._listener_owned_by", return_value=True) as owned:
            yield owned

    def test_our_proxy_is_accepted(self) -> None:
        with patch("urllib.request.urlopen", _answer({"k-1": 200})):
            assert _wait_for_health(4000, "k-1", _alive(), timeout=2) is True

    def test_squatter_accepting_any_key_is_rejected(self) -> None:
        anything = _answer({})
        anything.side_effect = None
        anything.return_value.__enter__.return_value.status = 200
        with patch("urllib.request.urlopen", anything):
            assert _wait_for_health(4000, "k-1", _alive(), timeout=2) is False

    def test_server_rejecting_our_key_is_rejected(self) -> None:
        with patch("urllib.request.urlopen", _answer({})):
            assert _wait_for_health(4000, "k-1", _alive(), timeout=2) is False

    def test_exited_child_fails_fast(self) -> None:
        proc = MagicMock()
        proc.poll.return_value = 1
        with patch("urllib.request.urlopen", _answer({"k-1": 200})):
            assert _wait_for_health(4000, "k-1", proc, timeout=2) is False

    def test_foreign_listener_never_sees_the_key(self, owned: MagicMock) -> None:
        owned.return_value = False
        urlopen = _answer({"k-1": 200})
        with patch("urllib.request.urlopen", urlopen):
            assert _wait_for_health(4000, "k-1", _alive(), timeout=2) is False
        sent = [_bearer(c.args[0]) for c in urlopen.call_args_list]
        assert sent
        assert not any(sent)

    @pytest.mark.parametrize("swap", [True, False])
    def test_fallback_randomizes_probe_order(self, owned: MagicMock, swap: bool) -> None:
        owned.return_value = None
        urlopen = _answer({"k-1": 200})
        with (
            patch("urllib.request.urlopen", urlopen),
            patch("mulder.orchestrator.proxy.secrets.choice", return_value=swap) as choice,
        ):
            assert _wait_for_health(4000, "k-1", _alive(), timeout=2) is True
        choice.assert_called_once()
        keyed = [_bearer(c.args[0]) for c in urlopen.call_args_list if _bearer(c.args[0])]
        assert len(keyed) == 2
        assert (keyed[0] == "k-1") is not swap

    def test_start_reports_failure_without_key(self) -> None:
        pm = ProxyManager(models=[MODEL])
        key = pm.env_overrides["ANTHROPIC_AUTH_TOKEN"]
        with (
            patch("shutil.which", return_value="/usr/local/bin/litellm"),
            patch("mulder.orchestrator.proxy.litellm_reasoning", return_value={}),
            patch("mulder.orchestrator.proxy._free_port", return_value=4000),
            patch("mulder.orchestrator.proxy._wait_for_health", return_value=False),
            patch("subprocess.Popen"),
            pytest.raises(RuntimeError) as exc,
        ):
            pm.start()
        assert key not in str(exc.value)
