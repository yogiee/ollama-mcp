#!/usr/bin/env python3
"""
maintenance.py — Sync, benchmark, and update registry.json for ollama-local MCP.
Run manually after pulling or removing Ollama models.
"""
import argparse
import ast
import copy
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ollama

from manifest import write_manifest

# ── Paths & constants ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
REGISTRY_PATH = BASE_DIR / "registry.json"
CONFIG_PATH = BASE_DIR / "config.json"
BENCH_IMAGE = BASE_DIR / "assets" / "bench_image.jpg"

_IMAGE_GEN_MODELS = {"x/flux2-klein:9b", "x/z-image-turbo:latest"}
_DEFAULT_THRESHOLD = 0.5

_EMPTY_BENCHMARKS: dict = {
    "json_compliance":    {"score": None, "ms_p50": None, "tested": None},
    "html_generation":    {"score": None, "ms_p50": None, "tested": None},
    "code_python":        {"score": None, "ms_p50": None, "tested": None},
    "instruction_follow": {"score": None, "tested": None},
    "tool_call_format":   {"score": None, "tested": None},
    "vision_basic":       {"score": None, "ms_p50": None, "tested": None},
    "latency_baseline":   {"ms_p50": None, "tested": None},
}

# tool → (required_capability, benchmark_names_used_for_scoring)
_TOOL_SPEC: dict[str, tuple[str, list[str]]] = {
    "local_chat":   ("completion", ["instruction_follow", "json_compliance"]),
    "local_code":   ("completion", ["code_python", "html_generation", "json_compliance"]),
    "local_vision": ("vision",     ["vision_basic"]),
    "local_ocr":    ("vision",     ["vision_basic"]),
    "local_embed":  ("embedding",  []),
}

QUICK_TESTS = ["json_compliance", "latency_baseline"]
FULL_TESTS = [
    "json_compliance", "html_generation", "code_python",
    "instruction_follow", "tool_call_format", "vision_basic", "latency_baseline",
]

# ── Registry I/O ───────────────────────────────────────────────────────────────
def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        sys.exit(f"ERROR: registry.json not found at {REGISTRY_PATH}")
    with REGISTRY_PATH.open() as f:
        return json.load(f)


def save_registry(registry: dict) -> None:
    with REGISTRY_PATH.open("w") as f:
        json.dump(registry, f, indent=2)
    print(f"Saved: {REGISTRY_PATH}")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open() as f:
        return json.load(f)


# ── Small utilities ────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_image_gen(name: str) -> bool:
    return name in _IMAGE_GEN_MODELS


def _is_benched(entry: dict) -> bool:
    return any(
        isinstance(v, dict) and v.get("score") is not None
        for v in entry.get("benchmarks", {}).values()
    )


def _model_skeleton(size_bytes: int, capabilities: list[str]) -> dict:
    return {
        "size_gb": round(size_bytes / 1e9, 1),
        "ollama_capabilities": capabilities,
        "benchmarks": copy.deepcopy(_EMPTY_BENCHMARKS),
        "avoid_for": [],
        "notes": "",
    }


# ── Ollama helpers ─────────────────────────────────────────────────────────────
def _get_capabilities(client: ollama.Client, model: str) -> list[str]:
    try:
        resp = client.show(model)
        caps = getattr(resp, "capabilities", None)
        if caps:
            return [str(c) for c in caps]
    except Exception:
        pass
    return ["completion"]


def _get_loaded_models(client: ollama.Client) -> list[str]:
    try:
        ps = getattr(client, "ps", None)
        if ps is None:
            return []
        resp = ps()
        models = getattr(resp, "models", [])
        return [m.model for m in models] if models else []
    except Exception:
        return []


# ── Benchmark tests ────────────────────────────────────────────────────────────
def _timed_chat(client: ollama.Client, model: str, messages: list, **kwargs) -> tuple[str, float]:
    t0 = time.monotonic()
    resp = client.chat(model=model, messages=messages, **kwargs)
    return resp.message.content, (time.monotonic() - t0) * 1000


def bench_json_compliance(client: ollama.Client, model: str) -> dict:
    prompt = (
        'Respond with only a JSON object. No explanation, no markdown fences. '
        'The object must have exactly these keys: "name" (string), "age" (integer), '
        '"tags" (array of strings). Example values are fine.'
    )
    content, ms = _timed_chat(client, model, [{"role": "user", "content": prompt}])
    try:
        obj = json.loads(content.strip())
    except json.JSONDecodeError:
        return {"score": 0.0, "ms_p50": round(ms), "tested": _now()}
    required = {"name": str, "age": int, "tags": list}
    hits = sum(k in obj and isinstance(obj[k], t) for k, t in required.items())
    score = 1.0 if hits == 3 else 0.5 if hits >= 1 else 0.0
    return {"score": score, "ms_p50": round(ms), "tested": _now()}


def bench_html_generation(client: ollama.Client, model: str) -> dict:
    prompt = (
        'Output only raw HTML. No explanation, no markdown code fences. '
        'Create a simple HTML page with: a <head> containing a <title> with text "Test Page", '
        'and a <body> containing an <h1> with text "Hello" and a <p> with text "World".'
    )
    content, ms = _timed_chat(client, model, [{"role": "user", "content": prompt}])
    text = content.strip()
    if text.startswith("```"):
        return {"score": 0.0, "ms_p50": round(ms), "tested": _now()}
    tl = text.lower()
    checks = [
        "<head>" in tl,
        "<title>" in tl and "Test Page" in text,
        "<body>" in tl,
        "<h1>" in tl and "Hello" in text,
        "<p>" in tl and "World" in text,
    ]
    passed = sum(checks)
    score = 1.0 if passed == 5 else 0.5 if passed >= 3 else 0.0
    return {"score": score, "ms_p50": round(ms), "tested": _now()}


def bench_code_python(client: ollama.Client, model: str) -> dict:
    prompt = (
        "Output only Python code. No explanation, no markdown fences. "
        "Write a function called `add_numbers` that takes two arguments and returns their sum."
    )
    content, ms = _timed_chat(client, model, [{"role": "user", "content": prompt}])
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {"score": 0.0, "ms_p50": round(ms), "tested": _now()}
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    correct_name = any(f.name == "add_numbers" for f in funcs)
    correct_sig = any(f.name == "add_numbers" and len(f.args.args) == 2 for f in funcs)
    score = 1.0 if correct_sig else 0.5 if correct_name else 0.0
    return {"score": score, "ms_p50": round(ms), "tested": _now()}


def bench_instruction_follow(client: ollama.Client, model: str) -> dict:
    prompt = (
        "Respond with exactly 10 words. Count carefully before responding. "
        "Describe what the sky looks like."
    )
    content, _ = _timed_chat(client, model, [{"role": "user", "content": prompt}])
    diff = abs(len(content.strip().split()) - 10)
    score = 1.0 if diff == 0 else 0.7 if diff == 1 else 0.3 if diff == 2 else 0.0
    return {"score": score, "tested": _now()}


def bench_tool_call_format(client: ollama.Client, model: str) -> dict:
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    try:
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            tools=tools,
        )
        calls = getattr(resp.message, "tool_calls", None) or []
        if calls:
            fn = calls[0].function
            name_ok = getattr(fn, "name", None) == "get_weather"
            args = getattr(fn, "arguments", {}) or {}
            score = 1.0 if name_ok and "city" in args else 0.0
        else:
            score = 0.0
    except Exception:
        score = 0.0
    return {"score": score, "tested": _now()}


def bench_vision_basic(client: ollama.Client, model: str) -> dict:
    if not BENCH_IMAGE.exists():
        print(f"  SKIP vision_basic: {BENCH_IMAGE} not found")
        return {"score": None, "ms_p50": None, "tested": None}
    img = BENCH_IMAGE.read_bytes()
    t0 = time.monotonic()
    try:
        resp = client.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": "Describe what you see in this image in one sentence.",
                "images": [img],
            }],
        )
        ms = (time.monotonic() - t0) * 1000
        score = 1.0 if "apple" in resp.message.content.lower() else 0.0
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return {"score": 0.0, "ms_p50": None, "tested": _now()}
    return {"score": score, "ms_p50": round(ms), "tested": _now()}


def bench_latency_baseline(client: ollama.Client, model: str) -> dict:
    prompt = "Say the word 'ready' and nothing else."
    times = []
    for _ in range(3):
        _, ms = _timed_chat(client, model, [{"role": "user", "content": prompt}])
        times.append(ms)
    return {"ms_p50": round(statistics.median(times)), "tested": _now()}


_BENCH_FNS = {
    "json_compliance":    bench_json_compliance,
    "html_generation":    bench_html_generation,
    "code_python":        bench_code_python,
    "instruction_follow": bench_instruction_follow,
    "tool_call_format":   bench_tool_call_format,
    "vision_basic":       bench_vision_basic,
    "latency_baseline":   bench_latency_baseline,
}

# ── Benchmark orchestration ────────────────────────────────────────────────────
def _bench_one(registry: dict, client: ollama.Client, model: str, tests: list[str]) -> None:
    if model not in registry.get("models", {}):
        print(f"  {model}: not in registry — run --sync first")
        return
    entry = registry["models"][model]
    caps = entry.get("ollama_capabilities", [])
    print(f"\n  {model}")

    for test in tests:
        if test == "tool_call_format" and "tools" not in caps:
            print(f"    {test:<25} SKIP (no tools capability)")
            continue
        if test == "vision_basic" and "vision" not in caps:
            print(f"    {test:<25} SKIP (no vision capability)")
            continue
        if test in ("json_compliance", "html_generation", "code_python",
                    "instruction_follow", "latency_baseline") and "completion" not in caps:
            print(f"    {test:<25} SKIP (no completion capability)")
            continue

        print(f"    {test:<25} ", end="", flush=True)
        try:
            result = _BENCH_FNS[test](client, model)
            entry["benchmarks"][test] = result
            parts = []
            if result.get("score") is not None:
                parts.append(f"score={result['score']:.1f}")
            if result.get("ms_p50") is not None:
                parts.append(f"{result['ms_p50']}ms")
            print(" ".join(parts) if parts else "ok")
        except Exception as exc:
            print(f"ERROR: {exc}")


def cmd_bench(
    registry: dict,
    client: ollama.Client,
    target: Optional[str],
    new_only: bool,
    quick: bool,
) -> None:
    loaded = _get_loaded_models(client)

    if target:
        models_to_bench = [target]
    else:
        models_to_bench = sorted(registry.get("models", {}).keys())
        if new_only:
            models_to_bench = [
                m for m in models_to_bench
                if not _is_benched(registry["models"][m]) and m not in loaded
            ]

    if not models_to_bench:
        print("No models to benchmark.")
        return

    if loaded and not new_only:
        print(f"\nWARNING: {', '.join(loaded)} is currently loaded and running.")
        print("  Benchmarking will evict it and interrupt any queued tasks.")
        if input("  Continue? [y/N] ").strip().lower() != "y":
            sys.exit("Aborted.")

    tests = QUICK_TESTS if quick else FULL_TESTS
    print(f"\nRunning {'quick' if quick else 'full'} benchmarks on {len(models_to_bench)} model(s):")
    for model in models_to_bench:
        _bench_one(registry, client, model, tests)

    save_registry(registry)


# ── Scoring & apply ────────────────────────────────────────────────────────────
def _composite_score(entry: dict, bench_names: list[str], all_latencies: list[float]) -> float:
    benchmarks = entry.get("benchmarks", {})
    task_scores = [
        benchmarks[b]["score"]
        for b in bench_names
        if isinstance(benchmarks.get(b), dict) and benchmarks[b].get("score") is not None
    ]
    task = sum(task_scores) / len(task_scores) if task_scores else 0.0

    lat_entry = benchmarks.get("latency_baseline", {})
    ms = lat_entry.get("ms_p50") if isinstance(lat_entry, dict) else None
    lat = (min(all_latencies) / ms) if ms and all_latencies else 0.5

    return round(0.6 * task + 0.4 * lat, 4)


def cmd_apply(registry: dict, config: dict) -> None:
    config_overrides = config.get("tool_overrides", {})
    threshold = config.get("bench_threshold", _DEFAULT_THRESHOLD)
    models = registry.get("models", {})

    new_defaults: dict[str, Optional[str]] = {}
    new_avoid: dict[str, list[str]] = {n: list(e.get("avoid_for", [])) for n, e in models.items()}
    new_scores: dict[str, Optional[float]] = {}   # tool → winning score (for diff display)
    old_scores: dict[str, Optional[float]] = {}   # tool → current default's score
    new_ranked: dict[str, list] = {}              # tool → eligible models best-first (for fallbacks)

    for tool, (req_cap, bench_names) in _TOOL_SPEC.items():
        eligible = [
            n for n, e in models.items()
            if req_cap in e.get("ollama_capabilities", [])
        ]

        if not eligible:
            new_defaults[tool] = registry.get("tool_defaults", {}).get(tool)
            continue

        # Embedding tools: no scoring, just pick first eligible
        if not bench_names:
            new_defaults[tool] = config_overrides.get(tool, eligible[0])
            new_ranked[tool] = list(eligible)  # unscored; registry order is the best signal we have
            continue

        # Latency values for normalization
        latencies = [
            models[n]["benchmarks"]["latency_baseline"]["ms_p50"]
            for n in eligible
            if isinstance(models[n].get("benchmarks", {}).get("latency_baseline"), dict)
            and models[n]["benchmarks"]["latency_baseline"].get("ms_p50") is not None
        ]

        candidates = []
        for name in eligible:
            entry = models[name]
            bench = entry.get("benchmarks", {})
            task_scores = [
                bench[b]["score"]
                for b in bench_names
                if isinstance(bench.get(b), dict) and bench[b].get("score") is not None
            ]
            if not task_scores:
                continue  # unbenched

            avg_task = sum(task_scores) / len(task_scores)

            # Populate avoid_for for low scorers
            for b in bench_names:
                if isinstance(bench.get(b), dict) and bench[b].get("score") is not None:
                    if bench[b]["score"] < threshold and b not in new_avoid[name]:
                        new_avoid[name].append(b)

            if avg_task < threshold:
                continue

            candidates.append((name, _composite_score(entry, bench_names, latencies)))

        if not candidates:
            new_defaults[tool] = registry.get("tool_defaults", {}).get(tool)
            new_scores[tool] = None
            new_ranked[tool] = []
        else:
            candidates.sort(key=lambda x: x[1], reverse=True)
            winner, score = candidates[0]
            new_defaults[tool] = winner
            new_scores[tool] = score
            new_ranked[tool] = [name for name, _ in candidates]

        # Score for the current default (for diff display)
        current = registry.get("tool_defaults", {}).get(tool)
        if current and current in models:
            task_scores = [
                models[current]["benchmarks"][b]["score"]
                for b in bench_names
                if isinstance(models[current].get("benchmarks", {}).get(b), dict)
                and models[current]["benchmarks"][b].get("score") is not None
            ]
            old_scores[tool] = (
                _composite_score(models[current], bench_names, latencies)
                if task_scores else None
            )
        else:
            old_scores[tool] = None

    # Carry forward manually-managed defaults not computed here (e.g. local_image.*,
    # which map image modes to image-gen models and are never benchmarked).
    for tool, model in registry.get("tool_defaults", {}).items():
        new_defaults.setdefault(tool, model)

    # config overrides always win
    for tool, model in config_overrides.items():
        new_defaults[tool] = model

    # Fallback chains for the manifest (best-first alternatives minus the effective primary),
    # so BenchLLAMA's drop-report protects models we'd re-pin to — see manifest.py.
    new_fallbacks = {
        tool: _fallback_chain(ranked, new_defaults.get(tool))
        for tool, ranked in new_ranked.items()
    }

    _print_apply_diff(
        registry.get("tool_defaults", {}), new_defaults,
        {n: e.get("avoid_for", []) for n, e in models.items()}, new_avoid,
        old_scores, new_scores,
    )

    if input("\nWrite changes? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    registry["tool_defaults"] = new_defaults
    registry["tool_fallbacks"] = new_fallbacks
    for name, avoid in new_avoid.items():
        if name in models:
            models[name]["avoid_for"] = avoid
    save_registry(registry)
    print(f"Manifest: {write_manifest(new_defaults, source=registry.get('defaults_source', 'manual'), fallbacks=new_fallbacks)}")


def _print_apply_diff(
    old_defaults: dict, new_defaults: dict,
    old_avoid: dict, new_avoid: dict,
    old_scores: dict, new_scores: dict,
) -> None:
    print("\ntool_defaults changes:")
    for tool in sorted(set(old_defaults) | set(new_defaults)):
        old = old_defaults.get(tool, "(none)")
        new = new_defaults.get(tool, "(none)")
        if old == new:
            print(f"  {tool:<20} {old}  (unchanged)")
        else:
            ns = new_scores.get(tool)
            os_ = old_scores.get(tool)
            score_info = ""
            if ns is not None and os_ is not None:
                score_info = f"  (score: {ns:.2f} vs {os_:.2f})"
            elif ns is not None:
                score_info = f"  (score: {ns:.2f})"
            print(f"  {tool:<20} {old}  →  {new}{score_info}")

    print("\navoid_for additions:")
    any_change = False
    for name in sorted(new_avoid):
        added = [b for b in new_avoid[name] if b not in (old_avoid.get(name) or [])]
        if added:
            print(f"  {name}: added {added}")
            any_change = True
    if not any_change:
        print("  No changes.")


# ── Sync ───────────────────────────────────────────────────────────────────────
def cmd_sync(registry: dict, client: ollama.Client) -> None:
    print("Syncing with Ollama...")
    try:
        all_models = client.list().models
    except Exception as exc:
        sys.exit(f"ERROR: Ollama not reachable: {exc}")

    ollama_models = {m.model: m for m in all_models}  # image-gen admitted to the 'image' lane
    registry.setdefault("models", {})

    registry_names = set(registry["models"].keys())
    ollama_names = set(ollama_models.keys())

    removed = registry_names - ollama_names
    new = ollama_names - registry_names

    for name in sorted(removed):
        del registry["models"][name]
        print(f"  REMOVED  {name}")

    for name in sorted(new):
        size_bytes = getattr(ollama_models[name], "size", 0) or 0
        # Image-gen models report no usable chat/vision capabilities; tag them for the
        # image lane (never benchmarked — local_image defaults are set manually).
        caps = ["image"] if _is_image_gen(name) else _get_capabilities(client, name)
        registry["models"][name] = _model_skeleton(size_bytes, caps)
        print(f"  NEW      {name}  [{', '.join(caps)}]")

    if not removed and not new:
        print("  No changes.")

    registry["last_synced"] = _now()
    save_registry(registry)


# ── Report ─────────────────────────────────────────────────────────────────────
def cmd_report(registry: dict, config: dict) -> None:
    overrides = config.get("tool_overrides", {})
    effective = {**registry.get("tool_defaults", {}), **overrides}

    print("=" * 64)
    print("REGISTRY REPORT")
    print(f"Last synced : {registry.get('last_synced') or 'never'}")
    print(f"Schema v    : {registry.get('schema_version', '?')}")
    print()

    print("Tool Defaults:")
    for tool, model in effective.items():
        tag = " (config override)" if tool in overrides else ""
        print(f"  {tool:<20} {model}{tag}")
    print()

    print("Models:")
    for name, entry in sorted(registry.get("models", {}).items()):
        caps = ", ".join(entry.get("ollama_capabilities", []))
        size = entry.get("size_gb") or "?"
        benched = _is_benched(entry)
        avoid = entry.get("avoid_for", [])
        status = "benched  " if benched else "UNBENCHED"
        print(f"  [{status}] {name}  ({size} GB, {caps})")
        if avoid:
            print(f"             avoid_for: {avoid}")
        if benched:
            for test, result in entry.get("benchmarks", {}).items():
                if not isinstance(result, dict):
                    continue
                score = result.get("score")
                ms = result.get("ms_p50")
                if score is None and ms is None:
                    continue
                parts = []
                if score is not None:
                    parts.append(f"score={score:.1f}")
                if ms is not None:
                    parts.append(f"{ms}ms")
                print(f"             {test:<25} {' '.join(parts)}")
    print("=" * 64)

    unbenched = [n for n, e in registry.get("models", {}).items() if not _is_benched(e)]
    if unbenched:
        print(f"\nUnbenched models ({len(unbenched)}): {', '.join(unbenched)}")
        print("  Run: python maintenance.py --bench --new")


# ── Reset ──────────────────────────────────────────────────────────────────────
def cmd_reset(registry: dict, model: str) -> None:
    if model not in registry.get("models", {}):
        sys.exit(f"ERROR: '{model}' not in registry. Run --sync first.")
    registry["models"][model]["benchmarks"] = copy.deepcopy(_EMPTY_BENCHMARKS)
    registry["models"][model]["avoid_for"] = []
    print(f"Reset benchmarks for {model}")
    save_registry(registry)


# ── Import BenchLLAMA rankings ─────────────────────────────────────────────────
_BUS_RANKINGS = Path.home() / ".config" / "ollama-consumers" / "benchllama-rankings.json"

# OllamaMCP tool → BenchLLAMA ranking list. vision/ocr/embed lists are purpose-ordered, so the
# top installed model is the right pick. workers/coders are quality-ordered — the data-current
# baseline is the top installed, while deliberate efficiency picks (trading rank for size/speed/
# consistency, which is a human judgment the data can't settle) live as config overrides.
# local_image.* are unmapped → manual, carried forward untouched.
_TOOL_RANKING = {
    "local_chat":   "workers",
    "local_code":   "coders",
    "local_vision": "vision",
    "local_ocr":    "vision_fast_ocr",
    "local_embed":  "embedding_long",
}

# How many fallback models to publish per tool in the consumer manifest. The fallback chain is the
# models we'd re-pin to if the primary were dropped; BenchLLAMA protects primaries + fallbacks from
# its drop-report. Keep it short — a long chain protects the whole fleet and defeats the report.
FALLBACK_DEPTH = 2


def _fallback_chain(ranked_installed: list, primary: Optional[str],
                    depth: int = FALLBACK_DEPTH) -> list:
    """Top-`depth` installed alternatives (best-first) for a tool, excluding the effective primary."""
    return [m for m in ranked_installed if m != primary][:depth]


def cmd_import_benchllama(registry: dict, config: dict, path: Optional[str]) -> None:
    rankings_path = Path(path) if path else _BUS_RANKINGS
    if not rankings_path.exists():
        sys.exit(f"ERROR: BenchLLAMA rankings not found: {rankings_path}\n"
                 "Run BenchLLAMA's export, or pass the path explicitly.")
    data = json.loads(rankings_path.read_text())
    rk = data.get("rankings", {})
    by_model = {m["name"]: m for m in data.get("models", [])}
    generated = (data.get("generated") or "")[:10]
    source = f"benchllama@{generated}" if generated else "benchllama"

    installed = set(registry.get("models", {}).keys())
    overrides = config.get("tool_overrides", {})

    new_defaults = dict(registry.get("tool_defaults", {}))  # preserves manual keys (local_image.*)
    top_pick: dict[str, Optional[str]] = {}
    for tool, lst in _TOOL_RANKING.items():
        ranked = rk.get(lst, [])
        pick = next((m for m in ranked if m in installed), None)
        top_pick[tool] = pick
        if pick:
            new_defaults[tool] = pick
        else:
            print(f"WARNING: no installed model in rankings.{lst} for {tool}; "
                  f"keeping {new_defaults.get(tool)!r}")  # never silently reroute (invariant #5)

    # Deliberate efficiency overrides (config.json) always win — invariant #3.
    for tool, model in overrides.items():
        new_defaults[tool] = model

    # Fallback chains for the manifest: top installed alternatives per tool, minus the effective
    # primary. Published so BenchLLAMA's drop-report protects models we'd re-pin to (see manifest.py).
    new_fallbacks: dict[str, list] = {}
    for tool, lst in _TOOL_RANKING.items():
        ranked_installed = [m for m in rk.get(lst, []) if m in installed]
        new_fallbacks[tool] = _fallback_chain(ranked_installed, new_defaults.get(tool))

    _print_import_diff(registry.get("tool_defaults", {}), new_defaults,
                       top_pick, overrides, rk, by_model, source, new_fallbacks)

    if input("\nWrite changes? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return
    registry["tool_defaults"] = new_defaults
    registry["tool_fallbacks"] = new_fallbacks
    registry["defaults_source"] = source
    save_registry(registry)
    print(f"Manifest: {write_manifest(new_defaults, source=source, fallbacks=new_fallbacks)}")


def _print_import_diff(old, new, top_pick, overrides, rk, by_model, source, fallbacks=None) -> None:
    fallbacks = fallbacks or {}

    def stat(name: str) -> str:
        m = by_model.get(name, {})
        return f"{m.get('disk_gb', '?')}GB {m.get('tps', '?')}tps"

    print(f"\ntool_defaults from {source}:")
    for tool, lst in _TOOL_RANKING.items():
        ranked = rk.get(lst, [])
        eff = new.get(tool)
        was = old.get(tool, "(none)")
        rank = f"{lst} #{ranked.index(eff) + 1}" if eff in ranked else f"{lst} (off-list)"
        override_note = ""
        if overrides.get(tool) == eff and top_pick.get(tool) != eff:
            override_note = f"  (config override; rankings-top {top_pick.get(tool)})"
        change = "  (unchanged)" if was == eff else f"  ⟵ {was}"
        print(f"  {tool:<13} {eff}  [{rank}, {stat(eff)}]{override_note}{change}")
        fb = fallbacks.get(tool, [])
        if fb:
            print(f"  {'':<13} └ fallbacks: {', '.join(fb)}")
    for tool in sorted(t for t in new if t not in _TOOL_RANKING):
        print(f"  {tool:<13} {new[tool]}  [manual]")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Maintain ollama-local MCP registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python maintenance.py --sync\n"
            "  python maintenance.py --bench --new\n"
            "  python maintenance.py --bench qwen3.5:9b-q4_K_M\n"
            "  python maintenance.py --bench --quick\n"
            "  python maintenance.py --report\n"
            "  python maintenance.py --apply\n"
            "  python maintenance.py --import-benchllama\n"
            "  python maintenance.py --reset qwen3.5:9b-q4_K_M --bench qwen3.5:9b-q4_K_M\n"
        ),
    )
    parser.add_argument("--sync", action="store_true",
                        help="Sync registry with current Ollama model list")
    parser.add_argument("--bench", nargs="?", const="__all__", metavar="MODEL",
                        help="Benchmark models (all, or a specific model)")
    parser.add_argument("--new", action="store_true",
                        help="With --bench: only unbenched models (skips loaded ones)")
    parser.add_argument("--quick", action="store_true",
                        help="With --bench: latency + JSON compliance only")
    parser.add_argument("--report", action="store_true",
                        help="Print registry status, scores, and warnings")
    parser.add_argument("--apply", action="store_true",
                        help="Recompute tool defaults from benchmarks (shows diff, asks confirmation)")
    parser.add_argument("--import-benchllama", nargs="?", const="__bus__", metavar="PATH",
                        dest="import_benchllama",
                        help="Set tool defaults from BenchLLAMA's rankings.json "
                             "(default: the shared ~/.config/ollama-consumers bus)")
    parser.add_argument("--reset", metavar="MODEL",
                        help="Clear all benchmark data for a model")
    args = parser.parse_args()

    if not any([args.sync, args.bench is not None, args.report, args.apply,
                args.import_benchllama is not None, args.reset]):
        parser.print_help()
        sys.exit(0)

    config = load_config()
    client = ollama.Client(
        host=config.get("ollama_host", "http://localhost:11434")
    )

    if args.reset:
        registry = load_registry()
        cmd_reset(registry, args.reset)

    if args.sync:
        registry = load_registry()
        cmd_sync(registry, client)

    if args.bench is not None:
        registry = load_registry()
        target = None if args.bench == "__all__" else args.bench
        cmd_bench(registry, client, target, args.new, args.quick)

    if args.report:
        registry = load_registry()
        cmd_report(registry, config)

    if args.apply:
        registry = load_registry()
        cmd_apply(registry, config)

    if args.import_benchllama is not None:
        registry = load_registry()
        path = None if args.import_benchllama == "__bus__" else args.import_benchllama
        cmd_import_benchllama(registry, config, path)


if __name__ == "__main__":
    main()
