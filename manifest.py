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


def _assignment(tool: str, model: str) -> dict:
    """Wrap a tool→model pick as a schema-2 assignment object."""
    capability, basis = _CAPABILITY_MAP.get(tool, ("manual", None))
    entry = {"model": model, "capability": capability}
    if basis:
        entry["basis"] = basis
    entry["tier"] = "primary"
    return entry


def build_manifest(tool_defaults: dict, source: str = "manual") -> dict:
    """Build the schema-2 manifest from a tool→model mapping (config overrides already merged in)."""
    assignments = {tool: _assignment(tool, model) for tool, model in tool_defaults.items() if model}
    models_in_use = sorted({model for model in tool_defaults.values() if model})
    return {
        "schema": 2,
        "consumer": CONSUMER,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection_policy": SELECTION_POLICY,
        "source": source,
        "assignments": assignments,
        "models_in_use": models_in_use,
    }


def write_manifest(tool_defaults: dict, source: str = "manual") -> Path:
    """Write the manifest to the conventional path and return it."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(build_manifest(tool_defaults, source), indent=2) + "\n")
    return MANIFEST_PATH
