# Ollama Local MCP — Model Inventory

Current as of: 2026-05-27
Host: localhost:11434

## Model List

| Model | Size | Capabilities | Notes |
|-------|------|-------------|-------|
| `nomic-embed-text:latest` | 274 MB | embedding | Dedicated embedding model — default for `local_embed` |
| `qwen3.5:2b-q4_K_M` | 1.9 GB | completion, vision, tools, thinking | Fastest general model — good for quick cheap tasks |
| `qwen3.5:9b-q4_K_M` | 6.6 GB | completion, vision, tools, thinking | Balanced speed/quality — initial default for `local_chat` |
| `glm-4.7-flash:latest` | 19 GB | completion, tools, thinking | Heavy reasoning, no vision |
| `qwen3-vl:8b` | 6.1 GB | completion, vision, tools, thinking | Vision-specialized — initial default for `local_vision` |
| `devstral-small-2:latest` | 15 GB | completion, vision, tools | Mistral coding model — initial default for `local_code` |
| `llama3.2-vision:11b` | 7.8 GB | completion, vision | Vision capable, no tools |
| `x/flux2-klein:9b` | 11 GB | image | Image generation — handled by existing `ollama-images` skill |
| `gpt-oss:latest` | 13 GB | completion, tools, thinking | Tools + thinking, no vision |
| `glm-ocr:latest` | 2.2 GB | completion, vision, tools | OCR-specialized — initial default for `local_ocr` |
| `x/z-image-turbo:latest` | 12 GB | image | Image generation — handled by existing `ollama-images` skill |
| `gemma3n:latest` | 7.5 GB | completion | No tools/thinking, general completion |
| `llama3.1:latest` | 4.9 GB | completion, tools | Reliable fallback, no vision |
| `gemma4:e2b` | 7.2 GB | completion, vision, audio, tools, thinking | Multimodal incl. audio |
| `gemma4:31b` | 19 GB | completion, vision, tools, thinking | Highest quality locally available |
| `gemma4:latest` | 9.6 GB | completion, vision, audio, tools, thinking | Multimodal incl. audio |

---

## Initial Tool Default Assumptions

These are starting assumptions before benchmarks are run. Run `--bench` and `--apply`
to replace with data-driven defaults.

| Tool | Initial Default | Rationale |
|------|----------------|-----------|
| `local_chat` | `qwen3.5:9b-q4_K_M` | Fast, tools+thinking, good balance |
| `local_code` | `devstral-small-2:latest` | Mistral's coding-specialized model |
| `local_vision` | `qwen3-vl:8b` | Vision-specialized, reasonable size |
| `local_ocr` | `glm-ocr:latest` | OCR-specialized, small and fast |
| `local_embed` | `nomic-embed-text:latest` | Only embedding model available |

---

## Models to Exclude from General MCP Tools

| Model | Reason |
|-------|--------|
| `x/flux2-klein:9b` | Image generation only — handled by `ollama-images` skill |
| `x/z-image-turbo:latest` | Image generation only — handled by `ollama-images` skill |

---

## Interesting Candidates for Specific Use Cases

### `gemma4:31b`
Highest quality locally. Slow (~19 GB to load). Good as a `model=` override when
quality matters more than speed. Vision + tools + thinking.

### `qwen3.5:2b-q4_K_M`
Fastest chat model. 1.9 GB — loads instantly, likely already resident. Good for
bulk/cheap tasks where latency matters (e.g. classifying a list of items).

### `gemma4:e2b` / `gemma4:latest`
Both have audio capability — potential future `local_audio` tool if there's a use case.
Not exposed in initial MCP tool set.

### `glm-ocr:latest`
Only 2.2 GB. Purpose-built for OCR. Should be the `local_ocr` default unless benchmarks
show otherwise.

### `llama3.1:latest`
The known-good fallback from prior projects (HTML/JSON work). If Qwen or Gemma models
fail benchmark tests, Llama 3.1 is a reliable fallback for `local_chat` and `local_code`.

---

## Notes on Known Failure Modes

- **Qwen models + HTML/JSON generation:** Observed in prior project work — Qwen kept
  failing structured output tasks where Llama models worked. This is exactly what the
  `html_generation` and `json_compliance` benchmark tests are designed to catch.
  Do not assume Ollama capability flags (`tools`) imply reliable structured output.

- **Model eviction:** Ollama loads one model at a time by default. Loading a large model
  (gemma4:31b, glm-4.7-flash, devstral) evicts whatever was previously loaded. Be aware
  of this when using `local_code` (devstral at 15 GB) during sessions where another
  large model is active.

---

## Recommended Models to Consider Pulling (Not Yet Installed)

These are suggestions — pull only if a specific use case warrants them:

| Model | Use Case |
|-------|---------|
| `mxbai-embed-large` | Higher quality embeddings than nomic if RAG accuracy matters |
| `deepseek-coder-v2:16b` | Alternative coding model to compare against devstral in benchmarks |
| `llava:13b` | Alternative vision model if qwen3-vl benchmarks poorly |
