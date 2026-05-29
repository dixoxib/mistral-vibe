# Project Ideas

## Context Continuity for Compaction

After compaction, the message chain retains everything that carries semantic weight:

```
[system, user1..5, seam1…seamX, summary1, user6..10, seamX+1…seamY, summary2, user11..15, seamY+1…seamZ, summary3]
```

- **User messages**: original questions — preserved so the LLM sees exact user intent
- **Seams**: structured checkpoints every ~200k tokens — active goal, constraints, decisions, open questions
- **Summaries**: compaction outputs — distilled state of everything before them

The chain is progressive: each summary encapsulates all prior state. Seams between summaries provide structured navigation points. Together they form a lossy but navigable context archive — the LLM can traverse from recent to ancient by reading seams.

Neither seams nor summaries should be filtered out during compaction. Everything stays.
