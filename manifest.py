"""Consumer manifest writer.

Publishes OllamaMCP's *effective* tool→model assignments to a conventional path so other
local tools — chiefly BenchLLAMA's usage-aware drop-report — can see what this consumer uses
and how it chose. Shared contract: every Ollama-consuming server writes one JSON file to
``~/.config/ollama-consumers/<consumer>.json``; a reader globs ``*.json`` in that directory.

File-only by design (no MCP tool): the drop-report is a plain script and shouldn't depend on a
live Claude session to invoke another server's tool. Writing is Ollama-independent (pure config),
so it succeeds even when Ollama is down.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

CONSUMER = "ollama-local"
SELECTION_POLICY = "efficiency-balanced"  # vs MemoryCentral's "requirements-fit"
MANIFEST_DIR = Path.home() / ".config" / "ollama-consumers"
MANIFEST_PATH = MANIFEST_DIR / f"{CONSUMER}.json"

# tool key → (capability, basis) for schema-2 assignments. `capability` drives BenchLLAMA's
# per-capability drop logic (see the bus README's capability vocabulary); `basis` names the
# exact ranking list when a capability has more than one (embedding short vs long). Keys not
# listed here (e.g. local_image.*) fall back to "manual" — never benchmarked, protected from drop.
_CAPABILITY_MAP: dict[str, tuple[str, Optional[str]]] = {
    "local_chat":   ("chat", None),
    "local_code":   ("coding", None),
    "local_vision": ("vision", None),
    "local_ocr":    ("ocr", None),
    "local_embed":  ("embedding", "embedding_long"),
}


def _assignment(base_tool: str, model: str, tier: str = "primary") -> dict:
    """Wrap a tool→model pick as a schema-2 assignment object.

    `base_tool` is the real tool key (drives capability/basis) even for fallback entries, whose
    dict key carries a ``#fallbackN`` suffix — so a fallback inherits its primary's capability and
    BenchLLAMA protects it per-capability rather than as a capability-less ``models_in_use`` entry.
    """
    capability, basis = _CAPABILITY_MAP.get(base_tool, ("manual", None))
    entry = {"model": model, "capability": capability}
    if basis:
        entry["basis"] = basis
    entry["tier"] = tier
    return entry


def build_manifest(tool_defaults: dict, source: str = "manual",
                   fallbacks: Optional[dict] = None) -> dict:
    """Build the schema-2 manifest from a tool→model mapping (config overrides already merged in).

    `fallbacks` maps a tool key → ordered list of fallback models (best-first) — the models this
    consumer would re-pin to if the primary were dropped. Each becomes its own ``tier:"fallback"``
    assignment (keyed ``<tool>#fallbackN``) sharing the primary's capability, and joins the flat
    ``models_in_use`` protected set. BenchLLAMA's drop-report protects the union of primaries +
    fallbacks per capability, so a model in our fallback chain is never drop-recommended out from
    under us. Keep the chain short — listing the whole fleet would protect everything and make the
    drop-report useless.
    """
    fallbacks = fallbacks or {}
    assignments: dict[str, dict] = {}
    protected: list[str] = []
    for tool, model in tool_defaults.items():
        if not model:
            continue
        assignments[tool] = _assignment(tool, model, "primary")
        protected.append(model)
        seen = {model}
        rank = 0
        for fb in fallbacks.get(tool, []):
            if not fb or fb in seen:
                continue
            rank += 1
            seen.add(fb)
            assignments[f"{tool}#fallback{rank}"] = _assignment(tool, fb, "fallback")
            protected.append(fb)
    return {
        "schema": 2,
        "consumer": CONSUMER,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection_policy": SELECTION_POLICY,
        "source": source,
        "assignments": assignments,
        "models_in_use": sorted(set(protected)),  # flat protected set: primaries + fallbacks
    }


def write_manifest(tool_defaults: dict, source: str = "manual",
                   fallbacks: Optional[dict] = None) -> Path:
    """Write the manifest to the conventional path and return it."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(build_manifest(tool_defaults, source, fallbacks), indent=2) + "\n"
    )
    return MANIFEST_PATH
