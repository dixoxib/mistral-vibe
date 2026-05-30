from __future__ import annotations

from typing import Any

import pytest

from vibe.core.config import ModelConfig
from vibe.core.llm.backend.generic import SEAM_MARKER, OpenAIAdapter
from vibe.core.types import LLMMessage, Role


def _tool(name: str, content: str) -> LLMMessage:
    return LLMMessage(role=Role.tool, content=content, name=name, tool_call_id="x")


def _user(content: str) -> LLMMessage:
    return LLMMessage(role=Role.user, content=content)


def _assistant(content: str) -> LLMMessage:
    return LLMMessage(role=Role.assistant, content=content)


def _seam(text: str = " summary") -> LLMMessage:
    return LLMMessage(role=Role.user, content=SEAM_MARKER + text, injected=True)


class TestSeamPruning:
    def test_no_prune_idx_does_nothing(self) -> None:
        messages = [_user("hello"), _assistant("hi"), _tool("read_file", "contents")]
        converted = _prepare(messages, prune_idx=None)
        assert converted[2]["content"] == "contents"

    def test_prune_idx_prunes_tools_before_index(self) -> None:
        messages = [
            _user("hi"),
            _tool("old", "old result"),
            _seam(),
            _tool("recent", "recent result"),
        ]
        converted = _prepare(messages, prune_idx=2)
        assert converted[1]["content"] == "[tool result for old pruned]"
        assert converted[3]["content"] == "recent result"

    def test_prune_idx_zero_prunes_nothing(self) -> None:
        messages = [_tool("a", "first"), _user("hi")]
        converted = _prepare(messages, prune_idx=0)
        assert converted[0]["content"] == "first"

    def test_config_defaults(self) -> None:
        cfg = ModelConfig(name="test", provider="test", alias="test")
        assert cfg.seam_interval == 0
        assert cfg.seam_prune_margin == 0


class TestSeamInjection:
    @pytest.mark.asyncio
    async def test_injects_filled_seam_when_threshold_exceeded(self) -> None:
        from tests.conftest import build_test_agent_loop
        from tests.mock.utils import mock_llm_chunk
        from tests.stubs.fake_backend import FakeBackend
        from tests.test_agent_stats import make_config

        summary_content = "### Active Goal\nBuild a thing"
        backend = FakeBackend([
            [mock_llm_chunk(content="first response")],
            [mock_llm_chunk(content=summary_content)],
            [mock_llm_chunk(content="after seam")],
        ])

        config = make_config(input_price=0.0, output_price=0.0)
        config.models[0].seam_interval = 1
        config.models[0].seam_prune_margin = 64_000

        agent = build_test_agent_loop(config=config, backend=backend)
        agent._last_seam_chars = 0

        async for _ in agent.act("hello"):
            pass

        seams = [
            m for m in agent.messages
            if m.role == Role.user and m.injected and SEAM_MARKER in (m.content or "")
        ]
        assert len(seams) == 1
        assert summary_content in seams[0].content

    @pytest.mark.asyncio
    async def test_skips_when_below_threshold(self) -> None:
        from tests.conftest import build_test_agent_loop
        from tests.mock.utils import mock_llm_chunk
        from tests.stubs.fake_backend import FakeBackend
        from tests.test_agent_stats import make_config

        backend = FakeBackend([mock_llm_chunk(content="response")])
        config = make_config(input_price=0.0, output_price=0.0)
        config.models[0].seam_interval = 1_000_000
        config.models[0].seam_prune_margin = 64_000

        agent = build_test_agent_loop(config=config, backend=backend)
        agent._last_seam_chars = 0

        async for _ in agent.act("hello"):
            pass

        seams = [
            m for m in agent.messages
            if m.role == Role.user and m.injected and SEAM_MARKER in (m.content or "")
        ]
        assert len(seams) == 0

    @pytest.mark.asyncio
    async def test_temp_messages_removed_after_injection(self) -> None:
        from tests.conftest import build_test_agent_loop
        from tests.mock.utils import mock_llm_chunk
        from tests.stubs.fake_backend import FakeBackend
        from tests.test_agent_stats import make_config

        summary_content = "### Active Goal\nTest"
        backend = FakeBackend([
            [mock_llm_chunk(content="first response")],
            [mock_llm_chunk(content=summary_content)],
            [mock_llm_chunk(content="after seam")],
        ])

        config = make_config(input_price=0.0, output_price=0.0)
        config.models[0].seam_interval = 1
        config.models[0].seam_prune_margin = 64_000

        agent = build_test_agent_loop(config=config, backend=backend)
        agent._last_seam_chars = 0

        async for _ in agent.act("hello"):
            pass

        messages = list(agent.messages)
        last_msg = messages[-1]
        assert last_msg.role == Role.assistant
        assert last_msg.content == "after seam"

        second_last = messages[-2]
        assert second_last.role == Role.user
        assert second_last.injected
        assert SEAM_MARKER in (second_last.content or "")


def _prepare(
    messages: list[LLMMessage], *, prune_idx: int | None
) -> list[dict[str, Any]]:
    adapter = OpenAIAdapter()
    converted = []
    for i, msg in enumerate(messages):
        dumped = msg.model_dump(
            exclude_none=True,
            exclude={
                "message_id",
                "reasoning_message_id",
                "reasoning_state",
                "injected",
            },
        )
        if msg.role == "assistant" and "reasoning_content" not in dumped:
            dumped["reasoning_content"] = ""
        if prune_idx is not None and i < prune_idx and msg.role == "tool":
            name = dumped.get("name", "tool")
            dumped["content"] = f"[tool result for {name} pruned]"
        converted.append(adapter._reasoning_to_api(dumped, "reasoning_content"))
    return converted
