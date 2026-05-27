# Ollama Local MCP — Architecture Spec

## MCP Server: `ollama-local`

Transport: **stdio** (no persistent port, no daemon — Claude spawns it on demand via
`mcpServers` config in `~/.claude/settings.json`).

### Registration (settings.json)

```json
{
  "mcpServers": {
    "ollama-local": {
      "command": "python",
      "args": ["/path/to/ollama-mcp/server.py"],
      "env": {
        "OLLAMA_HOST": "http://localhost:11434"
      }
    }
  }
}
```

---

## Tools

Each tool maps to a task category. The model that actually runs is read from
`registry.json → tool_defaults` at server startup. All tools accept an optional
`model` parameter to override the default.

### `local_chat`

General-purpose reasoning, drafts, summarization, cheap Q&A.

```
local_chat(
  prompt: str,
  system?: str,          # optional system prompt
  model?: str,           # overrides registry default
  thinking?: bool        # enable extended thinking if model supports it (default: false)
) → str
```

Default model: `qwen3.5:9b-q4_K_M`

---

### `local_code`

Code generation, review, refactoring, debugging.

```
local_code(
  prompt: str,
  language?: str,        # hint for the model (e.g. "python", "typescript")
  model?: str
) → str
```

Default model: `devstral-small-2:latest`

---

### `local_vision`

Analyze images, screenshots, diagrams. Accepts a file path or URL.

```
local_vision(
  image_path: str,       # local path or http(s) URL
  prompt: str,           # what to analyze / ask about the image
  model?: str
) → str
```

Default model: `qwen3-vl:8b`

---

### `local_ocr`

Extract text from documents, receipts, scans. Optimized for text-dense images.

```
local_ocr(
  image_path: str,
  model?: str
) → str
```

Default model: `glm-ocr:latest`

---

### `local_embed`

Generate embeddings for semantic search, similarity, or RAG pipelines.

```
local_embed(
  input: str | list[str],
  model?: str
) → list[float] | list[list[float]]
```

Default model: `nomic-embed-text:latest`

---

### `list_local_models`

Introspection — returns current Ollama model list with capabilities and registry status.

```
list_local_models() → list[{name, capabilities, in_registry, benchmarked, tool_default_for}]
```

No parameters. Calls `GET /api/tags` and cross-references with registry.

---

## Registry JSON Schema

```json
{
  "schema_version": 1,
  "last_synced": "2026-05-27T10:00:00Z",

  "tool_defaults": {
    "local_chat":   "qwen3.5:9b-q4_K_M",
    "local_code":   "devstral-small-2:latest",
    "local_vision": "qwen3-vl:8b",
    "local_ocr":    "glm-ocr:latest",
    "local_embed":  "nomic-embed-text:latest"
  },

  "models": {
    "qwen3.5:9b-q4_K_M": {
      "size_gb": 6.6,
      "ollama_capabilities": ["completion", "vision", "tools", "thinking"],
      "benchmarks": {
        "json_compliance":    { "score": null, "ms_p50": null, "tested": null },
        "html_generation":    { "score": null, "ms_p50": null, "tested": null },
        "code_python":        { "score": null, "ms_p50": null, "tested": null },
        "instruction_follow": { "score": null, "tested": null },
        "tool_call_format":   { "score": null, "tested": null },
        "latency_baseline":   { "ms_p50": null, "tested": null }
      },
      "avoid_for": [],
      "notes": ""
    }
  }
}
```

**Notes on `avoid_for`:** Populated by `--apply` when a model scores below threshold
(default: 0.5) on a category. The MCP server logs a warning if a tool is called with
a model that has that tool's task category in `avoid_for`.

---

## Server Startup Behavior

1. Read `registry.json` — load `tool_defaults` into memory
2. Verify Ollama is reachable (`GET /api/tags`) — if not, tools still register but
   return a clear error on invocation
3. If a tool's default model is not present in current Ollama model list, log a
   warning and fall back to the first available model with matching capability
4. Register all tools with the MCP SDK

## Error Handling Philosophy

- Model not found → clear error message naming the missing model + hint to run `--sync`
- Ollama unreachable → surface that directly, don't silently fail
- Timeout (configurable, default 120s) → return timeout error with model name
- Never silently fall back to a random model — always be explicit about what ran

## Config Overrides (`config.json`)

User-maintained file that takes precedence over registry-computed defaults.
Useful for pinning a specific model regardless of benchmark scores.

```json
{
  "tool_overrides": {
    "local_code": "gemma4:31b"
  },
  "ollama_host": "http://localhost:11434",
  "timeout_seconds": 120,
  "bench_threshold": 0.5
}
```

`config.json` is never written by the maintenance script — it's always user-controlled.
