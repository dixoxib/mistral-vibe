#!/usr/bin/env python3
"""Remove empty seam templates from a Vibe session file."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SEAM_MARKER = "## Context Seam"
EMPTY_TEMPLATE_CHARS = 878  # length of the unfilled seam template


def clean_session(path: Path) -> int:
    with open(path) as f:
        lines = f.readlines()

    kept = []
    removed = 0
    for line in lines:
        d = json.loads(line.strip())
        if (
            d.get("role") == "user"
            and d.get("injected")
            and SEAM_MARKER in str(d.get("content", ""))
            and len(str(d.get("content", ""))) == EMPTY_TEMPLATE_CHARS
        ):
            removed += 1
            continue
        kept.append(line)

    if removed:
        with open(path, "w") as f:
            f.writelines(kept)

        meta_path = path.parent / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            non_system = sum(
                1 for line in kept
                if json.loads(line.strip()).get("role") != "system"
            )
            meta["total_messages"] = non_system
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

    return removed


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / ".vibe/logs/session"
    if target.is_dir():
        sessions = sorted(target.iterdir(), reverse=True)
        target = sessions[0] / "messages.jsonl" if sessions else None
        if not target or not target.exists():
            print("No session found")
            sys.exit(1)

    n = clean_session(Path(target))
    print(f"Removed {n} empty seam template(s) from {target}")
