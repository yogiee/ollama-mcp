# ollama-mcp

Run local Ollama models as tools inside [Claude Code](https://claude.ai/code). Offload drafts, code generation, vision tasks and embeddings to your local GPU — no API calls, no data leaving your machine. On macOS 27+, an optional native layer adds on-device image generation and editing through Apple's Core AI.

```
Claude Code  →  local_chat / local_code / local_vision / …  →  Ollama   →  your models
             →  local_image (macOS 27+, optional)           →  Core AI  →  FLUX.2 [klein]
```

The MCP server registers under the name **`ollama-mcp`** (matching the repo/folder).

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
.venv/bin/pip install -r requirements.txt
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
claude mcp add --scope user ollama-mcp \
  -e OLLAMA_HOST=http://localhost:11434 \
  -- /path/to/ollama-mcp/.venv/bin/python /path/to/ollama-mcp/server.py
```

Replace `/path/to/ollama-mcp` with the absolute path where you cloned this repo. `--scope user` makes the server available in every Claude Code project, not just the current one.

Verify the server connected:

```bash
claude mcp list
# ollama-mcp: ... ✓ Connected
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
| `local_image` | Image generation and editing (macOS 27+ only, see below) | `prompt`, optional `mode` (`"photo"`/`"design"`), `edit_image`, `seed`, `steps`, `model`, `output_dir` |
| `list_local_models` | List installed models and registry status | — |

All tools accept an optional `model` parameter to override the registry default for that call.

`local_image` picks its model from `mode`: `"photo"` (photorealism — portraits, landscapes, product shots) or `"design"` (text-in-image, logos, UI mockups, posters). Pass `edit_image` (a path or URL) to change an existing image instead, with the prompt describing the change; pass the previous result's path to chain edits. The same `seed` reproduces an image exactly. Output is 1024×1024 PNG, saved to `./Generated_images` in the current project (or `~/Pictures/Generated_images`), and the reply ends with the saved path. It is never benchmarked — see [Model recommendations](#model-recommendations) below.

---

## Model recommendations

These are the current defaults on this machine, computed by `maintenance.py` from benchmark scores (or, for `local_image`, set manually). They are written into `registry.json` under `tool_defaults` — yours will differ based on which models you have pulled.

| Tool | Default model |
|------|--------------|
| `local_chat` | `gemma4:12b-mlx` |
| `local_code` | `granite4.1:3b` |
| `local_vision` | `qwen2.5vl:3b` |
| `local_ocr` | `minicpm-v4.6:1b` |
| `local_embed` | `embeddinggemma:300m` |
| `local_image` (`mode="photo"`) | `coreai/flux2-klein-4b` |
| `local_image` (`mode="design"`) | `coreai/flux2-klein-4b` |

The `local_image.*` defaults are flat keys in `tool_defaults`, set manually and preserved across `--apply` (text-to-image has no objective structural grade to benchmark). Both modes use FLUX.2 [klein] 4B for now; a dedicated photoreal model is being evaluated. All other defaults are recomputed from benchmarks. Override any of them per call with `model=`, or pin them in `config.json`.

> <sub>Defaults above are based on current testing and usage on an Apple M1 Max (10-core CPU / 32-core GPU) + 32 GB RAM system. Faster or larger machines may favor bigger models; constrained machines may need smaller ones. Re-run `--bench` + `--apply` on your own hardware to get fleet-appropriate picks.</sub>

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

## Optional: on-device image generation (macOS 27+)

Ollama removed its image-generation engine in v0.32.6. Until it returns, `local_image` runs on Apple's **Core AI** through a small Swift helper in `native/`. This layer is optional and macOS-only: on any other system, or until the helper is built, the server simply starts without `local_image`. Everything else stays Ollama-only and cross-platform.

**Requirements:** macOS 27+ on Apple silicon, Xcode 27 (Swift 6.4), about 9 GB of free memory while an image is being made.

```bash
# 1. Build the helper (one-off, ~1–2 min; fetches apple/coreai-models at a pinned revision)
swift build -c release --package-path native

# 2. Put a Core AI diffusion bundle in the models folder
#    default: ~/Library/Application Support/OllamaMCP/models/coreai/<bundle>/
#    e.g. FLUX.2 [klein] 4B, exported with apple/coreai-models' `coreai.diffusion.export black-forest-labs/FLUX.2-klein-4B`

# 3. Record it in the registry, then restart Claude Code (or reconnect the server)
.venv/bin/python maintenance.py --sync
```

Bundles are discovered by their `metadata.json`; FLUX.2 and Sana Sprint bundles are supported, and their model ids look like `coreai/flux2-klein-4b`. Two optional `config.json` keys point elsewhere:

```json
{
  "native_models_dir":   "~/path/to/coreai/bundles",
  "native_image_helper": "~/path/to/ollamamcp-imagegen"
}
```

Measured on an M1 Max (32 GB) with FLUX.2 [klein] 4B: ~22 s per image, ~47 s per edit, peak ~7 GB. The very first run prepares the model for this machine (~70 s) and leaves a ~3.8 GB cache in `~/Library/Caches/coreai-cache/<macOS build>/ollamamcp-imagegen/`. If memory is short before an image, the server unloads its own idle Ollama models first; models other apps have loaded are left alone and named in the reply.

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

## Advanced: BenchLLAMA integration

If you run **BenchLLAMA** (a shared local benchmark/ranking tool), `ollama-mcp` can source its tool defaults from BenchLLAMA's cross-consumer rankings instead of its own benchmarks, and it publishes a consumer manifest so BenchLLAMA's drop-report knows which models you actually use. This is entirely optional — skip it if you don't use BenchLLAMA.

See **[docs/benchllama.md](docs/benchllama.md)**.

---

## Troubleshooting

**`Ollama not reachable`** — Ollama isn't running. Start it with `ollama serve`.

**Model not found error** — The registry references a model that was removed. Run `--sync` then `--apply`.

**`local_image` missing from the tool list** — it only registers on macOS 27+ with the helper built. Build it with `swift build -c release --package-path native`, then reconnect the server.

**`has no Core AI bundle`** — the image model's bundle isn't in the models folder. Check `native_models_dir` in `config.json`, then run `--sync`.

**Tools not showing in Claude Code** — Make sure you used `--scope user` when registering. Restart Claude Code after adding a new MCP server.

**Server not connecting** — Verify the path to `.venv/bin/python` and `server.py` are absolute and correct: `claude mcp get ollama-mcp`.
