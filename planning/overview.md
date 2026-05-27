# Ollama Local MCP — Project Overview

## What We're Building

A local MCP (Model Context Protocol) server that exposes Ollama-hosted models as tools
inside Claude Code sessions. Claude can offload tasks to local models — code review,
OCR, vision analysis, embeddings, cheap reasoning — without burning Claude API tokens
or sending data externally.

## Goals

- Let Claude delegate subtasks to appropriate local models transparently
- Minimize wasted compute from routing tasks to models that fail them
- Survive model changes in Ollama without manual code edits
- Keep the interface simple and debuggable for the caller (Claude or user)

## Key Decisions

### Explicit Routing over Smart Routing

**Decision:** Named tools with hardcoded-default models, not a smart router.

**Why:** Smart routing adds failure modes without real benefit here. Two flavors were
considered and rejected:

- **Rule/tag-based routing** — fragile, edge cases are silent failures, maintenance burden
- **Meta-LLM routing** — adds 2–5s latency per call, wakes a second model (resource
  pressure), classification errors are invisible, complexity roughly doubles

The clean insight: routing already happens at the tool-selection layer. Claude picks
`local_code` vs `local_vision` based on the task — that's the routing decision, and it
belongs there. The MCP server's job is execution, not re-routing.

**Tradeoff accepted:** Claude needs to know the tool lineup. Mitigated by clear tool
names and good descriptions in the MCP server.

### Registry-Driven Defaults

**Decision:** Tool-to-model mappings live in `registry.json`, not in server code.

**Why:** Decouples "what tool Claude calls" (stable, explicit) from "which model backs
that tool" (changes as models are added/removed/benchmarked). The MCP server reads the
registry at startup. A separate maintenance CLI writes to it.

This means model changes in Ollama never require touching the MCP server code — just
run the maintenance script and `--apply`.

### Benchmark-Informed Defaults

**Decision:** Tool defaults are computed from benchmark scores, not guessed.

**Why:** Real failure case from prior work — Qwen models kept failing HTML/JSON
generation tasks where Llama models worked fine. Capability flags from Ollama
(`tools`, `vision`, `thinking`) don't tell you output format reliability or latency.
Only running tests reveals that.

The maintenance script benchmarks each model across task categories and `--apply`
rewrites `tool_defaults` in the registry based on scores.

## What This Is Not

- Not a smart router or AI-powered dispatcher
- Not a persistent daemon (MCP server runs as stdio, spawned by Claude on demand)
- Not a replacement for Claude — local models handle cheap/specialized subtasks
- Not fully automated — maintenance is manually triggered, not on a cron

## Project Structure (target)

```
ollama-mcp/
├── server.py            # MCP server — reads registry, exposes tools to Claude
├── maintenance.py       # CLI — syncs, benchmarks, updates registry
├── benchmarks/          # Test suite definitions
│   ├── json_test.py
│   ├── html_test.py
│   ├── code_test.py
│   ├── instruction_test.py
│   ├── toolcall_test.py
│   ├── vision_test.py
│   └── latency_test.py
├── registry.json        # Auto-maintained model registry + benchmark results
├── config.json          # User overrides (always use X for Y, pins, etc.)
└── assets/              # Static test fixtures (test image for vision bench, etc.)
    └── bench_image.jpg
```

## Stack

- **Language:** Python (consistent with existing `ollama_image_gen.py` skill)
- **MCP SDK:** `mcp` Python library (stdio transport)
- **Ollama client:** `ollama` Python library
- **Registry:** Plain JSON file — human-readable, git-trackable
