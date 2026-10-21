"""Tests that fatal-error detection only trusts the SDK's structured error channel.

Analyst prose routinely quotes evidence such as sshd ``Permission denied
(publickey)`` lines or cloud ``AccessDeniedException`` strings. Only an
assistant message the CLI flagged with ``error`` (or an SDK exception) may
abort the investigation as an auth or model failure.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest
from claude_agent_sdk.types import AssistantMessage, TextBlock

from mulder.orchestrator.errors import AuthenticationError, ModelNotAvailableError
from mulder.orchestrator.models import ModelConfig
from mulder.orchestrator.session import SessionExecutor

_EVIDENCE_PROSE = [
    "Multiple Permission denied (publickey) entries precede the successful root login.",
    "CloudTrail shows AccessDeniedException for the attacker role before escalation.",
    "The dropped loader logs 'model not found' when its config is missing.",
    "Strings include 'invalid api key' and 'Not logged in · Please run /login'.",
    "The phishing kit says the model is not available on your account; you could try using it.",
]

_MODEL_ERROR = (
    "The model claude-sonnet-4-6@20250514 is not available on your "
    "vertex deployment. You could try using claude-haiku-4-5@20250414 instead."
)


def _make_session() -> SessionExecutor:
    """Create a SessionExecutor with a mocked dashboard."""
    return SessionExecutor(
        dashboard=MagicMock(),
        model_config=ModelConfig(),
        cwd="/tmp",
        env={},
        effort="max",
    )


def _assistant(text: str, error: str | None = None) -> AssistantMessage:
    """Build a real SDK AssistantMessage carrying one text block."""
    return AssistantMessage(
        content=[TextBlock(text=text)],
        model="claude-test",
        error=error,
        message_id="msg-1",
        usage={"input_tokens": 0, "output_tokens": 0},
    )


def _query_yielding(message: AssistantMessage):  # type: ignore[no-untyped-def]
    """Return a fake ``query`` that yields a single message."""

    async def mock_query(**kwargs: object) -> AsyncIterator[AssistantMessage]:
        yield message

    return mock_query


async def _execute(session: SessionExecutor, model: str = "claude-test") -> object:
    return await session.execute(
        system_prompt="test",
        prompt="test",
        model=model,
        allowed_tools=[],
        disallowed_tools=[],
        max_turns=1,
    )


class TestProseIsNotFatal:
    """Model prose quoting error phrases never aborts the run."""

    @pytest.mark.asyncio()
    @pytest.mark.parametrize("text", _EVIDENCE_PROSE)
    async def test_execute_prose_does_not_raise(self, text: str) -> None:
        session = _make_session()
        with patch("mulder.orchestrator.session.query", _query_yielding(_assistant(text))):
            await _execute(session)

    @pytest.mark.asyncio()
    @pytest.mark.parametrize("text", _EVIDENCE_PROSE)
    async def test_utility_prose_does_not_raise(self, text: str) -> None:
        session = _make_session()
        with patch("mulder.orchestrator.session.query", _query_yielding(_assistant(text))):
            result = await session.execute_utility(
                prompt="test", allowed_tools=[], label="test-utility"
            )
        assert result is None


class TestStructuredErrorsStillFatal:
    """Assistant messages the CLI flags with ``error`` still abort the run."""

    @pytest.mark.asyncio()
    async def test_execute_auth_error_raises(self) -> None:
        session = _make_session()
        msg = _assistant("Not logged in · Please run /login", error="authentication_failed")
        with (
            patch("mulder.orchestrator.session.query", _query_yielding(msg)),
            pytest.raises(AuthenticationError) as exc_info,
        ):
            await _execute(session)
        assert exc_info.value.suggestion

    @pytest.mark.asyncio()
    async def test_execute_auth_flag_without_known_text_raises(self) -> None:
        session = _make_session()
        msg = _assistant("API Error: 401", error="authentication_failed")
        with (
            patch("mulder.orchestrator.session.query", _query_yielding(msg)),
            pytest.raises(AuthenticationError),
        ):
            await _execute(session)

    @pytest.mark.asyncio()
    async def test_execute_model_error_raises(self) -> None:
        session = _make_session()
        msg = _assistant(_MODEL_ERROR, error="invalid_request")
        with (
            patch("mulder.orchestrator.session.query", _query_yielding(msg)),
            pytest.raises(ModelNotAvailableError) as exc_info,
        ):
            await _execute(session, model="claude-sonnet-4-6@20250514")
        assert exc_info.value.alternative == "claude-haiku-4-5@20250414"

    @pytest.mark.asyncio()
    async def test_utility_auth_error_raises(self) -> None:
        session = _make_session()
        msg = _assistant("Not logged in · Please run /login", error="authentication_failed")
        with (
            patch("mulder.orchestrator.session.query", _query_yielding(msg)),
            pytest.raises(AuthenticationError),
        ):
            await session.execute_utility(prompt="test", allowed_tools=[], label="test-utility")

    @pytest.mark.asyncio()
    async def test_utility_model_error_raises(self) -> None:
        session = _make_session()
        msg = _assistant(_MODEL_ERROR, error="invalid_request")
        with (
            patch("mulder.orchestrator.session.query", _query_yielding(msg)),
            pytest.raises(ModelNotAvailableError),
        ):
            await session.execute_utility(prompt="test", allowed_tools=[], label="test-utility")

    @pytest.mark.asyncio()
    async def test_sdk_exception_still_classified(self) -> None:
        session = _make_session()

        async def mock_query(**kwargs: object) -> AsyncIterator[AssistantMessage]:
            raise Exception("Not logged in · Please run /login")  # noqa: TRY002
            yield  # pragma: no cover

        with (
            patch("mulder.orchestrator.session.query", mock_query),
            pytest.raises(AuthenticationError),
        ):
            await _execute(session)
