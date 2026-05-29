from __future__ import annotations

from typing import Any

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
        assert cfg.seam_interval == 200_000
        assert cfg.seam_prune_margin == 64_000


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
