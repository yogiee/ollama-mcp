# BenchLLAMA integration

> **Optional / advanced.** This is only relevant if you also run **BenchLLAMA**, a shared
> local tool that benchmarks and ranks your Ollama models across every tool that consumes them.
> If you don't use BenchLLAMA, ignore this — `ollama-mcp` benchmarks and assigns models on its
> own (see [Model recommendations](../README.md#model-recommendations) and
> [How model assignment works](../README.md#how-model-assignment-works) in the main README).

`ollama-mcp` participates in a shared `~/.config/ollama-consumers/` bus. The integration has two
halves: it can **read** BenchLLAMA's rankings to pick models, and it **writes** a manifest so
BenchLLAMA can see what it uses.

---

## Reading: `--import-benchllama`

Set tool defaults from BenchLLAMA's shared `rankings.json` instead of running local benchmarks:

```bash
# Read the shared bus (~/.config/ollama-consumers)
.venv/bin/python maintenance.py --import-benchllama

# Or point at a specific rankings file
.venv/bin/python maintenance.py --import-benchllama /path/to/rankings.json
```

For each tool, it picks the highest-ranked model that is actually installed. Safeguards:

- **Manual `local_image.*` keys are preserved** — text-to-image has no objective structural grade,
  so those defaults are never recomputed.
- **`config.json` pins always win** — deliberate overrides are applied last (invariant #3).
- **Never silently reroutes** — if no installed model appears in a ranking list, it keeps the
  current default and prints a warning (invariant #5).

Like `--apply`, it prints a diff before writing.

---

## Writing: the consumer manifest

On startup and after `--apply` (and `--import-benchllama`), the server publishes its *effective*
tool→model assignments to:

```
~/.config/ollama-consumers/ollama-mcp.json
```

This is written by `manifest.py`. The shared contract: every Ollama-consuming tool drops one JSON
file in that directory, and readers — chiefly BenchLLAMA's usage-aware drop-report — glob `*.json`
to see which models are actually in use before recommending any be removed.

The manifest lists each tool's **primary** model plus a short **fallback chain** (the models this
consumer would re-pin to if its primary were dropped), so models you depend on are protected from
drop recommendations even when they aren't the current pick.

Writing the manifest is Ollama-independent (pure config), so it succeeds even when Ollama is down.

### Manifest shape

```json
{
  "schema": 2,
  "consumer": "ollama-mcp",
  "generated": "2026-06-24T00:00:00Z",
  "selection_policy": "efficiency-balanced",
  "source": "benchllama@2026-06-24",
  "assignments": {
    "local_chat":            { "model": "gemma4:latest",       "capability": "chat",   "tier": "primary" },
    "local_chat#fallback1":  { "model": "...",                 "capability": "chat",   "tier": "fallback" }
  },
  "models_in_use": ["..."]
}
```

- `assignments` — one entry per tool, plus `<tool>#fallbackN` entries that inherit the primary's
  `capability` so BenchLLAMA protects them per-capability.
- `models_in_use` — the flat protected set (primaries + fallbacks), deduped and sorted.
- `source` — `manual` on a plain startup write, or `benchllama@<date>` after an import.
