#!/usr/bin/env python3
"""Check seam spacing in a Vibe session file."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SEAM_MARKER = "## Context Seam"


def analyze_seams(path: Path) -> None:
    with open(path) as f:
        lines = f.readlines()

    seams = []
    total_chars = 0
    for i, line in enumerate(lines):
        d = json.loads(line.strip())
        content = d.get("content", "") or ""
        total_chars += len(content)
        if (
            d.get("role") == "user"
            and d.get("injected")
            and isinstance(content, str)
            and content.startswith(SEAM_MARKER)
        ):
            seams.append({
                "msg_index": i,
                "chars": total_chars,
                "tokens_est": total_chars // 4,
                "content_len": len(content),
            })

    if not seams:
        print("No seams found.")
        return

    print(f"Found {len(seams)} seam(s) in {len(lines)} messages "
          f"({total_chars:,} chars, ~{total_chars // 4:,} tokens)\n")

    prev_chars = 0
    for s in seams:
        gap_chars = s["chars"] - prev_chars
        gap_tokens = gap_chars // 4
        print(
            f"  msg #{s['msg_index']:>5}  "
            f"at ~{s['tokens_est']:>7,} tokens  "
            f"gap: {gap_tokens:>7,} tokens ({gap_chars:,} chars)  "
            f"size: {s['content_len']} chars"
            f"{' ← EMPTY TEMPLATE' if s['content_len'] < 1000 else ''}"
        )
        prev_chars = s["chars"]

    last = seams[-1]
    tail_chars = total_chars - last["chars"]
    print(f"\n  tail: {tail_chars // 4:,} tokens ({tail_chars:,} chars) from last seam to end")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / ".vibe/logs/session"
    if target.is_dir():
        sessions = sorted(target.iterdir(), reverse=True)
        target = sessions[0] / "messages.jsonl" if sessions else None
        if not target or not target.exists():
            print("No session found")
            sys.exit(1)

    analyze_seams(Path(target))
