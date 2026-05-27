#!/usr/bin/env python3
"""
ollama-local MCP server — exposes Ollama models as tools in Claude Code sessions.
Tool-to-model mappings are read from registry.json at startup.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import ollama
from mcp.server.fastmcp import FastMCP

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
REGISTRY_PATH = BASE_DIR / "registry.json"
CONFIG_PATH = BASE_DIR / "config.json"

# ── Load registry and config ───────────────────────────────────────────────────
def _load_json(path: Path, required: bool = False) -> dict:
    if not path.exists():
        if required:
            sys.exit(f"ERROR: {path.name} not found. Run: python maintenance.py --sync")
        return {}
    with path.open() as f:
        return json.load(f)


_registry = _load_json(REGISTRY_PATH, required=True)
_config = _load_json(CONFIG_PATH)

# config.tool_overrides always win over registry.tool_defaults
TOOL_DEFAULTS: dict[str, str] = {
    **_registry.get("tool_defaults", {}),
    **_config.get("tool_overrides", {}),
}
OLLAMA_HOST: str = _config.get(
    "ollama_host", os.environ.get("OLLAMA_HOST", "http://localhost:11434")
)

# ── Ollama client ──────────────────────────────────────────────────────────────
_client = ollama.Client(host=OLLAMA_HOST)

# ── Startup: verify reachability + warn on missing defaults ───────────────────
_available_models: list[str] = []
_ollama_reachable = False

try:
    _available_models = [m.model for m in _client.list().models]
    _ollama_reachable = True
    for _tool, _model in TOOL_DEFAULTS.items():
        if _model and _model not in _available_models:
            print(
                f"WARNING: {_tool} default model '{_model}' not found in Ollama. "
                "Run: python maintenance.py --sync",
                file=sys.stderr,
            )
except Exception as exc:
    print(f"WARNING: Ollama not reachable at {OLLAMA_HOST}: {exc}", file=sys.stderr)

# ── Helpers ────────────────────────────────────────────────────────────────────
def _resolve_model(tool: str, override: Optional[str]) -> tuple[bool, str]:
    """Return (ok, model_name) on success or (False, error_message) on failure."""
    if not _ollama_reachable:
        return False, f"Ollama not reachable at {OLLAMA_HOST}"
    m = override or TOOL_DEFAULTS.get(tool)
    if not m:
        return False, f"No model configured for {tool}. Run: python maintenance.py --sync"
    if m not in _available_models:
        return False, f"Model '{m}' not found in Ollama. Run: python maintenance.py --sync"
    return True, m


def _load_image(image_path: str) -> bytes:
    """Load image bytes from a local path or http(s) URL."""
    if image_path.startswith(("http://", "https://")):
        with urllib.request.urlopen(image_path, timeout=30) as resp:
            return resp.read()
    return Path(image_path).read_bytes()


# ── MCP server ─────────────────────────────────────────────────────────────────
mcp = FastMCP("ollama-local")


@mcp.tool()
def local_chat(
    prompt: str,
    system: Optional[str] = None,
    model: Optional[str] = None,
    thinking: bool = False,
) -> str:
    """General-purpose reasoning, drafts, summarization, and Q&A using a local Ollama model."""
    ok, result = _resolve_model("local_chat", model)
    if not ok:
        return f"Error: {result}"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = _client.chat(
            model=result,
            messages=messages,
            options={"think": True} if thinking else None,
        )
        return resp.message.content
    except Exception as exc:
        return f"Error: {exc}"


@mcp.tool()
def local_code(
    prompt: str,
    language: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Code generation, review, refactoring, and debugging using a local Ollama model."""
    ok, result = _resolve_model("local_code", model)
    if not ok:
        return f"Error: {result}"
    full_prompt = f"Language: {language}\n\n{prompt}" if language else prompt
    try:
        resp = _client.chat(
            model=result,
            messages=[{"role": "user", "content": full_prompt}],
        )
        return resp.message.content
    except Exception as exc:
        return f"Error: {exc}"


@mcp.tool()
def local_vision(
    image_path: str,
    prompt: str,
    model: Optional[str] = None,
) -> str:
    """Analyze an image, screenshot, or diagram. image_path can be a local file path or http(s) URL."""
    ok, result = _resolve_model("local_vision", model)
    if not ok:
        return f"Error: {result}"
    try:
        img = _load_image(image_path)
    except Exception as exc:
        return f"Error loading image '{image_path}': {exc}"
    try:
        resp = _client.chat(
            model=result,
            messages=[{"role": "user", "content": prompt, "images": [img]}],
        )
        return resp.message.content
    except Exception as exc:
        return f"Error: {exc}"


@mcp.tool()
def local_ocr(
    image_path: str,
    model: Optional[str] = None,
) -> str:
    """Extract text from a document, receipt, or scan. image_path can be a local file path or http(s) URL."""
    ok, result = _resolve_model("local_ocr", model)
    if not ok:
        return f"Error: {result}"
    try:
        img = _load_image(image_path)
    except Exception as exc:
        return f"Error loading image '{image_path}': {exc}"
    try:
        resp = _client.chat(
            model=result,
            messages=[{
                "role": "user",
                "content": "Extract all text from this image. Output the extracted text only, preserving layout where possible.",
                "images": [img],
            }],
        )
        return resp.message.content
    except Exception as exc:
        return f"Error: {exc}"


@mcp.tool()
def local_embed(
    input: str | list[str],
    model: Optional[str] = None,
) -> list:
    """Generate embeddings for semantic search or RAG. Returns a vector or list of vectors."""
    ok, result = _resolve_model("local_embed", model)
    if not ok:
        return [f"Error: {result}"]
    try:
        resp = _client.embed(model=result, input=input)
        return resp.embeddings
    except Exception as exc:
        return [f"Error: {exc}"]


@mcp.tool()
def list_local_models() -> list[dict]:
    """List all Ollama models with capabilities, registry status, and which tool each is the default for."""
    if not _ollama_reachable:
        return [{"error": f"Ollama not reachable at {OLLAMA_HOST}"}]
    registry_models: dict = _registry.get("models", {})
    default_for: dict[str, str] = {v: k for k, v in TOOL_DEFAULTS.items() if v}
    result = []
    for m in _client.list().models:
        name = m.model
        entry = registry_models.get(name, {})
        benchmarked = any(
            isinstance(v, dict) and v.get("score") is not None
            for v in entry.get("benchmarks", {}).values()
        )
        result.append({
            "name": name,
            "size_gb": entry.get("size_gb"),
            "capabilities": entry.get("ollama_capabilities", []),
            "in_registry": name in registry_models,
            "benchmarked": benchmarked,
            "tool_default_for": default_for.get(name),
            "avoid_for": entry.get("avoid_for", []),
        })
    return result


if __name__ == "__main__":
    mcp.run()
