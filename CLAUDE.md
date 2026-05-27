# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture at a Glance

```
server.py             — MCP server (stdio). Reads registry.json at startup. Exposes tools.
maintenance.py        — CLI for syncing, benchmarking, updating registry.json.
registry.json         — Auto-maintained. Maps tools to models + benchmark scores. Written by maintenance.py.
registry.json.example — Schema template. Copy to registry.json to start fresh.
config.json           — User overrides. Never written by maintenance.py. (gitignored)
assets/               — Static test fixtures (bench_image.jpg for vision benchmark)
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

| Tool | Task |
|------|------|
| `local_chat` | General reasoning, drafts, Q&A, summarization |
| `local_code` | Code generation, review, refactoring |
| `local_vision` | Image/screenshot/diagram analysis |
| `local_ocr` | Document/scan text extraction |
| `local_embed` | Embeddings for semantic search/RAG |
| `list_local_models` | Introspection — list installed models and their registry status |

Defaults are computed by `maintenance.py --apply` based on benchmark scores. All tools accept an optional `model` parameter to override the registry default.

---

## Setup

```bash
# 1. Create virtualenv and install dependencies
python3 -m venv .venv
.venv/bin/pip install mcp ollama

# 2. Initialize registry
cp registry.json.example registry.json

# 3. Sync installed Ollama models and benchmark them
.venv/bin/python maintenance.py --sync
.venv/bin/python maintenance.py --bench --new
.venv/bin/python maintenance.py --apply

# 4. Register with Claude Code (user-scope = available in all projects)
claude mcp add --scope user ollama-local \
  -e OLLAMA_HOST=http://localhost:11434 \
  -- /path/to/ollama-mcp/.venv/bin/python /path/to/ollama-mcp/server.py
```

Transport is stdio — no persistent port, no daemon. Claude spawns it on demand.

---

## Optional: config.json overrides

Create `config.json` to pin specific models regardless of benchmark scores:

```json
{
  "tool_overrides": {
    "local_chat":   "gemma4:e2b",
    "local_code":   "gemma4:e2b-mlx",
    "local_vision": "gemma4:e2b"
  }
}
```

---

## Maintenance Workflow

```bash
# After pulling a new model
.venv/bin/python maintenance.py --sync
.venv/bin/python maintenance.py --bench --new
.venv/bin/python maintenance.py --report
.venv/bin/python maintenance.py --apply

# After removing a model
.venv/bin/python maintenance.py --sync       # removes it from registry
.venv/bin/python maintenance.py --apply      # recomputes defaults without it

# Re-bench a specific model
.venv/bin/python maintenance.py --reset <model> && .venv/bin/python maintenance.py --bench <model>

# Quick health check (latency + JSON compliance only)
.venv/bin/python maintenance.py --bench --quick
```

`--apply` prints a diff and asks for confirmation before writing. Always review it.

---

## Stack

- Python 3.11+
- `mcp` — MCP SDK (stdio transport)
- `ollama` — Ollama Python client
- Requires Ollama running at `http://localhost:11434` (or set `OLLAMA_HOST`)

---

## Known Model Gotchas

- **Structured output is not guaranteed by capability flags** — some models advertise `tools` support but fail JSON/HTML output benchmarks. Always run `--bench` after pulling a new model.

- **Image-gen models are excluded** — models that only do image generation (no completion) are automatically skipped during sync.

- **Large model eviction** — Ollama loads one model at a time by default. Routing multiple tools to the same model minimizes memory churn.

- **MLX variants (Apple Silicon)** — MLX models are faster on M-series chips but are text-only (no vision). Use standard variants for `local_vision`.

---

## Git Workflow

Do not commit, push, or create PRs without explicit instruction.
Implement changes, report what's ready, then wait for "go ahead."
