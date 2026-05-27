# Ollama Local MCP — Maintenance CLI

## Purpose

The maintenance script keeps `registry.json` accurate as Ollama models change over
time. It is run manually — not on a cron, not automatically by the MCP server.

Typical triggers:
- After `ollama pull <new-model>`
- After `ollama rm <model>`
- When a model starts behaving unexpectedly (re-bench it)
- Periodically (monthly) to refresh latency baselines

---

## CLI Modes

```bash
# Check for model changes, update registry capabilities, flag new/removed
python maintenance.py --sync

# Benchmark all models not yet in registry (safe to run after pulling new models)
python maintenance.py --bench --new

# Full benchmark suite on all models
python maintenance.py --bench

# Benchmark one specific model
python maintenance.py --bench qwen3.5:9b-q4_K_M

# Print current registry: rankings, defaults, warnings, unbenched models
python maintenance.py --report

# Recompute tool_defaults from current benchmark scores and write to registry
python maintenance.py --apply

# Clear benchmark data for a model and re-run from scratch
python maintenance.py --reset qwen3.5:9b-q4_K_M --bench qwen3.5:9b-q4_K_M

# Quick health check — latency + JSON compliance only (1-2 min per model)
python maintenance.py --bench --quick
```

---

## Resource Awareness

Before starting any benchmark run, the script:

1. Calls `GET /api/ps` to check which model is currently loaded and running in Ollama
2. If a model is active, prints a warning:
   ```
   WARNING: glm-4.7-flash:latest is currently loaded and running.
   Benchmarking will evict it and interrupt any queued tasks.
   Continue? [y/N]
   ```
3. `--new` skips models that are currently loaded (won't interrupt active work)
4. Models are benched one at a time — no parallel benchmark runs

---

## Benchmark Test Suite

### Scoring

Each test returns a score from 0.0 to 1.0. Tests are deterministic — same prompt
every time, graded against an expected structure (not semantic quality). Tests are
fast by design: target 10–20 seconds per test per model.

---

### Test: `json_compliance`

**What it tests:** Can the model reliably output valid, schema-conforming JSON?
**Prompt:**
```
Respond with only a JSON object. No explanation, no markdown fences. 
The object must have exactly these keys: "name" (string), "age" (integer), "tags" (array of strings).
Example values are fine.
```
**Grading:**
- 1.0 — valid JSON, all required keys present, correct types
- 0.5 — valid JSON but missing keys or wrong types
- 0.0 — invalid JSON or non-JSON response

**Why this matters:** Many tasks (tool calls, structured output, data extraction) depend
on reliable JSON. This was the failure mode observed with Qwen on prior HTML/JSON work.

---

### Test: `html_generation`

**What it tests:** Can the model produce valid, structured HTML without hallucinating
extra content or mangling the structure?
**Prompt:**
```
Output only raw HTML. No explanation, no markdown code fences.
Create a simple HTML page with: a <head> containing a <title> with text "Test Page",
and a <body> containing an <h1> with text "Hello" and a <p> with text "World".
```
**Grading:**
- 1.0 — valid HTML, all required tags present with correct text content
- 0.5 — valid HTML structure but missing or wrong elements
- 0.0 — not valid HTML or wrapped in markdown fences

**Why this matters:** Direct failure case from prior project — Qwen failed this
consistently, Llama models worked.

---

### Test: `code_python`

**What it tests:** Can the model generate syntactically valid, callable Python?
**Prompt:**
```
Output only Python code. No explanation, no markdown fences.
Write a function called `add_numbers` that takes two arguments and returns their sum.
```
**Grading:**
- 1.0 — `ast.parse()` succeeds, function named correctly, callable with two args
- 0.5 — parseable but wrong function name or signature
- 0.0 — syntax error or non-code response

---

### Test: `instruction_follow`

**What it tests:** Does the model follow explicit, countable constraints?
**Prompt:**
```
Respond with exactly 10 words. Count carefully before responding.
Describe what the sky looks like.
```
**Grading:**
- 1.0 — exactly 10 words
- 0.7 — 9 or 11 words
- 0.3 — 8 or 12 words
- 0.0 — more than 2 words off

---

### Test: `tool_call_format`

**Applies to:** Models with `tools` in Ollama capabilities only.
**What it tests:** Does the model correctly format a tool call when given a tool schema?
Uses the Ollama `/api/chat` tools API with a simple `get_weather(city: str)` schema.
**Prompt:** "What's the weather in Tokyo?"
**Grading:**
- 1.0 — response contains a valid tool call with correct function name and argument
- 0.0 — no tool call, or malformed tool call JSON

---

### Test: `vision_basic`

**Applies to:** Models with `vision` in Ollama capabilities only.
**What it tests:** Basic image understanding — can the model describe a test image?
**Fixture:** `assets/bench_image.jpg` — a simple image with a clearly identifiable object
(e.g. a red apple on a white background). Ship this in the repo.
**Prompt:** "Describe what you see in this image in one sentence."
**Grading:**
- 1.0 — response contains the expected keyword (e.g. "apple")
- 0.0 — no relevant description or refusal

---

### Test: `latency_baseline`

**What it tests:** Time to complete a standard short prompt. Run 3 times, record p50.
**Prompt:** "Say the word 'ready' and nothing else."
**Output:** `ms_p50` — not a pass/fail score, stored as a measurement only.

---

## Scoring and `--apply` Logic

After benchmarks, `--apply` rewrites `tool_defaults` in `registry.json`:

1. For each tool category (chat, code, html, vision, ocr, embed):
   - Collect all models that have relevant capability
   - Filter out models with `score < bench_threshold` (default 0.5) for that category
   - Sort remaining by composite score (weighted: task relevance 60%, latency 40%)
   - Pick the top model as the new default
   - Add low-scoring models to their `avoid_for` list

2. `config.json` overrides are applied last — user pins always win.

3. Print a diff of what changed before writing:
   ```
   tool_defaults changes:
     local_code:  devstral-small-2:latest  →  devstral-small-2:latest  (unchanged)
     local_chat:  qwen3.5:9b-q4_K_M       →  gemma4:latest            (score: 0.91 vs 0.87)
   
   avoid_for additions:
     qwen3.5:9b-q4_K_M: added html_generation (score: 0.4)
   
   Write changes? [y/N]
   ```

---

## Suggested Workflow After Adding a New Model

```bash
ollama pull <new-model>
python maintenance.py --sync           # adds new model to registry skeleton
python maintenance.py --bench --new    # benchmarks only the new model
python maintenance.py --report         # review scores
python maintenance.py --apply          # update defaults if warranted
```

Total time for a mid-size model (~7–10 GB): approximately 5–8 minutes.
