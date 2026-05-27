# ollama-mcp

Run local Ollama models as tools inside [Claude Code](https://claude.ai/code). Offload drafts, code generation, vision tasks, and embeddings to your local GPU — no API calls, no data leaving your machine.

```
Claude Code  →  local_chat / local_code / local_vision  →  Ollama  →  your models
```

---

## Requirements

- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- At least one Ollama model pulled (e.g. `ollama pull gemma4:latest`)
- Python 3.11+
- [Claude Code](https://claude.ai/code) CLI

---

## Installation

### 1. Clone and install dependencies

```bash
git clone https://github.com/yogiee/ollama-mcp.git
cd ollama-mcp

python3 -m venv .venv
.venv/bin/pip install mcp ollama
```

### 2. Initialize the registry

```bash
cp registry.json.example registry.json
```

### 3. Sync and benchmark your models

```bash
# Discover all installed Ollama models
.venv/bin/python maintenance.py --sync

# Benchmark newly discovered models
.venv/bin/python maintenance.py --bench --new

# Preview the recommended tool → model assignments
.venv/bin/python maintenance.py --report

# Write the assignments to registry.json
.venv/bin/python maintenance.py --apply
```

`--apply` shows a diff and asks for confirmation before writing anything.

### 4. Register with Claude Code

```bash
claude mcp add --scope user ollama-local \
  -e OLLAMA_HOST=http://localhost:11434 \
  -- /path/to/ollama-mcp/.venv/bin/python /path/to/ollama-mcp/server.py
```

Replace `/path/to/ollama-mcp` with the absolute path where you cloned this repo. `--scope user` makes the server available in every Claude Code project, not just the current one.

Verify the server connected:

```bash
claude mcp list
# ollama-local: ... ✓ Connected
```

---

## Tools

Once registered, these tools are available in every Claude Code session:

| Tool | Default task | Accepts |
|------|-------------|---------|
| `local_chat` | Reasoning, drafts, Q&A, summarization | `prompt`, optional `system`, `thinking`, `model` |
| `local_code` | Code generation, review, refactoring | `prompt`, optional `language`, `model` |
| `local_vision` | Image / screenshot / diagram analysis | `image_path` (file or URL), `prompt`, optional `model` |
| `local_ocr` | Text extraction from documents or scans | `image_path` (file or URL), optional `model` |
| `local_embed` | Embeddings for semantic search / RAG | `input` (string or list), optional `model` |
| `list_local_models` | List installed models and registry status | — |

All tools accept an optional `model` parameter to override the registry default for that call.

---

## Optional: pin models with config.json

Create `config.json` in the project root to override the benchmark-computed defaults:

```json
{
  "tool_overrides": {
    "local_chat":   "gemma4:e2b",
    "local_code":   "gemma4:e2b",
    "local_vision": "gemma4:e2b"
  }
}
```

`config.json` is gitignored and never modified by `maintenance.py`. Your pins always win over registry defaults.

---

## Maintenance

```bash
# After pulling a new Ollama model
.venv/bin/python maintenance.py --sync
.venv/bin/python maintenance.py --bench --new
.venv/bin/python maintenance.py --apply

# After removing a model
.venv/bin/python maintenance.py --sync
.venv/bin/python maintenance.py --apply

# Re-run benchmarks for one model
.venv/bin/python maintenance.py --reset <model>
.venv/bin/python maintenance.py --bench <model>

# Quick health check (latency + JSON compliance only)
.venv/bin/python maintenance.py --bench --quick

# See current scores and assignments
.venv/bin/python maintenance.py --report
```

---

## How model assignment works

`--apply` scores each model on two dimensions:

- **Task score (60%)** — structural benchmarks: does JSON parse, are HTML tags present, does Python AST compile, does it follow instructions. Pass/fail, not semantic quality.
- **Latency score (40%)** — normalized against the fastest model in the fleet.

The model with the highest composite score for each tool's required capabilities becomes the default. If a model is missing from Ollama, the server returns a clear error rather than silently rerouting.

---

## Troubleshooting

**`Ollama not reachable`** — Ollama isn't running. Start it with `ollama serve`.

**Model not found error** — The registry references a model that was removed. Run `--sync` then `--apply`.

**Tools not showing in Claude Code** — Make sure you used `--scope user` when registering. Restart Claude Code after adding a new MCP server.

**Server not connecting** — Verify the path to `.venv/bin/python` and `server.py` are absolute and correct: `claude mcp get ollama-local`.
