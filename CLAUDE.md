# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Planning phase — no code has been written yet. The planning docs in the root directory are the source of truth for all design decisions:

- `planning/overview.md` — goals, key decisions (why explicit routing over smart routing, why registry-driven)
- `planning/architecture.md` — full tool specs, registry JSON schema, server startup behavior, config.json format
- `planning/maintenance.md` — CLI modes, benchmark test suite specs, scoring logic, `--apply` diff format
- `planning/models.md` — current Ollama model inventory with capabilities and initial default rationale

Read these before implementing. The decisions in `overview.md` are final unless explicitly revisited.

---

## Architecture at a Glance

```
server.py        — MCP server (stdio). Reads registry.json at startup. Exposes tools.
maintenance.py   — CLI for syncing, benchmarking, updating registry.json.
registry.json    — Auto-maintained. Maps tools to models + benchmark scores. Written by maintenance.py.
config.json      — User overrides. Never written by maintenance.py.
benchmarks/      — Test suite modules (json_test.py, html_test.py, code_test.py, etc.)
assets/          — Static test fixtures (bench_image.jpg for vision benchmark)
```

---

## Key Invariants — Do Not Break These

**1. Tool-to-model mappings live in `registry.json`, not in `server.py`.**
The MCP server reads `tool_defaults` from the registry at startup. Model names must not be hardcoded in server.py.

**2. No smart routing in the MCP server.**
The server does not inspect prompt content to decide which model to use. Claude picks the right tool (`local_code` vs `local_vision`) — that is the routing decision. The server's job is execution only.

**3. `config.json` is user-controlled. `maintenance.py` never writes to it.**
User pins in `config.json` always override registry-computed defaults.

**4. Benchmark tests must be deterministic and fast.**
Each test uses a fixed prompt and grades against structural criteria (JSON parses, HTML tags present, AST valid) — not semantic quality. Target: under 20 seconds per test per model.

**5. Never fall back to a random model silently.**
If a tool's default model is missing from Ollama, surface a clear error naming the missing model. Do not silently reroute.

---

## Tools Exposed by the MCP Server

| Tool | Default Model | Task |
|------|--------------|------|
| `local_chat` | qwen3.5:9b-q4_K_M | General reasoning, drafts, Q&A |
| `local_code` | devstral-small-2:latest | Code generation, review, refactoring |
| `local_vision` | qwen3-vl:8b | Image/screenshot analysis |
| `local_ocr` | glm-ocr:latest | Document/scan text extraction |
| `local_embed` | nomic-embed-text:latest | Embeddings for semantic search/RAG |
| `list_local_models` | — | Introspection |

All tools accept an optional `model` parameter to override the registry default.

---

## MCP Server Registration

Add to `~/.claude/settings.json` to enable in Claude Code sessions:

```json
{
  "mcpServers": {
    "ollama-local": {
      "command": "/path/to/ollama-mcp/.venv/bin/python",
      "args": ["/path/to/ollama-mcp/server.py"],
      "env": {
        "OLLAMA_HOST": "http://localhost:11434"
      }
    }
  }
}
```

Transport is stdio — no persistent port, no daemon. Claude spawns it on demand.

---

## Maintenance Workflow

```bash
# After pulling a new model
python maintenance.py --sync
python maintenance.py --bench --new
python maintenance.py --report
python maintenance.py --apply

# After removing a model
python maintenance.py --sync       # removes it from registry
python maintenance.py --apply      # recomputes defaults without it

# Re-bench a specific model
python maintenance.py --reset <model> && python maintenance.py --bench <model>

# Quick health check (latency + JSON compliance only)
python maintenance.py --bench --quick
```

`--apply` prints a diff and asks for confirmation before writing. Always review it.

---

## Stack

- Python 3.11+
- `mcp` — MCP SDK (stdio transport)
- `ollama` — Ollama Python client
- `registry.json` — plain JSON, human-readable, should be git-tracked

---

## Known Model Gotchas

- **Qwen models fail HTML/JSON structured output** — confirmed in prior work. The `html_generation` and `json_compliance` benchmarks exist specifically to catch this. Do not assume Ollama's `tools` capability flag implies reliable structured output.

- **Image-gen models (`flux2-klein`, `z-image-turbo`) are excluded** from MCP tools. They are handled by the separate `ollama-images` Claude Code skill.

- **Large model eviction** — Ollama loads one model at a time by default. `local_code` (devstral, 15 GB) will evict whatever was previously loaded. Warn the user in the `--bench` flow if a large model is currently active (`GET /api/ps`).

- **`llama3.1:latest` is the known-good fallback** — if Qwen or Gemma models fail benchmark tests, Llama 3.1 is a reliable fallback for `local_chat` and `local_code`.

---

## Git Workflow

Do not commit, push, or create PRs without explicit instruction.
Implement changes, report what's ready, then wait for "go ahead."
