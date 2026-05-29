from __future__ import annotations

from typing import Any

from textual.reactive import reactive

from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from vibe.core.types import AgentStats

_MILLION = 1_000_000
_THOUSAND = 1_000
_CENT_THRESHOLD = 0.01


def _fmt(n: int) -> str:
    if n >= _MILLION:
        return f"{n / _MILLION:.1f}M"
    if n >= _THOUSAND:
        return f"{n / _THOUSAND:.0f}k"
    return str(n)


def _fmt_cost(c: float) -> str:
    if c >= _CENT_THRESHOLD:
        return f"${c:.2f}"
    return f"${c:.4f}"


class ContextProgress(NoMarkupStatic):
    stats = reactive(AgentStats(), always_update=True)
    max_tokens = reactive(0)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def watch_stats(self, new: AgentStats) -> None:
        if self.max_tokens == 0:
            self.update("")
            return

        ctx_pct = min(100, new.context_tokens * 100 // self.max_tokens)

        last = (
            f"out:{_fmt(new.last_turn_completion_tokens)} "
            f"hit:{_fmt(new.last_turn_cache_hit_tokens)} "
            f"miss:{_fmt(new.last_turn_cache_miss_tokens)} "
            f"{_fmt_cost(new.last_turn_cost)}"
        )

        total = (
            f"out:{_fmt(new.session_completion_tokens)} "
            f"hit:{_fmt(new.session_cache_hit_tokens)} "
            f"miss:{_fmt(new.session_cache_miss_tokens)} "
            f"{_fmt_cost(new.session_cost)}"
        )

        self.update(
            f"{_fmt(new.context_tokens)}/{_fmt(self.max_tokens)} {ctx_pct}% | "
            f"{last} | \u2211 {total}"
        )

    def watch_max_tokens(self, new: int) -> None:
        self.watch_stats(self.stats)
