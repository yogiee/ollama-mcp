#!/usr/bin/env python3
"""
ollama-local MCP server — exposes Ollama models as tools in Claude Code sessions.
Tool-to-model mappings are read from registry.json at startup.
"""
import base64
import json
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

import ollama
from mcp.server.fastmcp import FastMCP

from manifest import write_manifest

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

# Publish effective assignments to the consumer manifest (file-only, Ollama-independent).
# Lets BenchLLAMA's drop-report see what this consumer uses — primaries AND fallback chains
# (persisted to the registry by maintenance.py) — without a live session.
try:
    write_manifest(
        TOOL_DEFAULTS,
        source=_registry.get("defaults_source", "manual"),
        fallbacks=_registry.get("tool_fallbacks"),
    )
except Exception as _exc:
    print(f"WARNING: could not write consumer manifest: {_exc}", file=sys.stderr)

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


# Image-gen modes → tool-default key. Claude picks the mode from the user's request
# (routing stays Claude-side, preserving the no-smart-routing invariant); the server
# only maps mode → model. "photo" → fast photorealism, "design" → text/UI/illustration.
_IMAGE_MODES = ("photo", "design")
_MLX_FIX = (
    "MLX library path issue. Run:\n"
    "  sudo ln -sf /opt/homebrew/lib/libmlxc.dylib /usr/local/lib/libmlxc.dylib\n"
    "  sudo ln -sf /opt/homebrew/lib/libmlx.dylib  /usr/local/lib/libmlx.dylib\n"
    "  brew services restart ollama"
)


def _image_output_dir(output_dir: Optional[str]) -> Path:
    """Resolve the save dir: explicit override → the project (cwd) folder → ~/Pictures fallback.

    The MCP server inherits Claude's working directory, so cwd is the project folder when
    launched in one. Falls back to ~/Pictures/Generated_images when cwd is not a usable project
    location (the home directory, filesystem root, or not writable).
    """
    if output_dir:
        return Path(output_dir)
    cwd = Path.cwd()
    if cwd not in (Path.home(), Path("/")) and os.access(cwd, os.W_OK):
        return cwd / "Generated_images"
    return Path.home() / "Pictures" / "Generated_images"


def _generate_image(
    model: str, prompt: str, width: int, height: int,
    steps: Optional[int], output_dir: Optional[str],
) -> str:
    """POST to Ollama /api/generate, decode the base64 image, save a PNG, return its path."""
    payload: dict = {
        "model": model, "prompt": prompt, "stream": False,
        "width": width, "height": height,
    }
    if steps:
        payload["steps"] = steps
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:  # first call loads the model
        data = json.loads(resp.read())

    if data.get("error"):
        msg = data["error"]
        if "libmlxc.dylib" in msg:
            return f"Error from Ollama: {msg}\n\nFIX: {_MLX_FIX}"
        return f"Error from Ollama: {msg}"

    image_b64 = data.get("image") or data.get("response") or ""
    if not image_b64:
        return f"Error: no image data in response (keys: {list(data.keys())})"

    out_dir = _image_output_dir(output_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:  # chosen dir unwritable after all → guaranteed-writable fallback
        out_dir = Path.home() / "Pictures" / "Generated_images"
        out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = model.split("/")[-1].replace(":", "-")
    safe = "".join(c if c.isalnum() or c in " _-" else "" for c in prompt[:45]).strip().replace(" ", "_")
    out_path = out_dir / f"{stamp}_{tag}_{safe}.png"
    out_path.write_bytes(base64.b64decode(image_b64))
    return str(out_path)


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
def local_image(
    prompt: str,
    mode: str = "photo",
    model: Optional[str] = None,
    width: int = 1024,
    height: int = 1024,
    steps: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Generate an image locally from a text prompt. Returns the saved PNG file path.

    mode="photo"  -> photorealism, portraits, landscapes, product shots (fast)
    mode="design" -> text-in-image, logos, UI mockups, posters, illustration, editing
    Pick mode from the user's request; pass model= to force a specific model.
    macOS-only; 1024x1024 default; the first call loads the model (~60-120s).
    Saves to ./Generated_images in the current project, else ~/Pictures/Generated_images.
    """
    if mode not in _IMAGE_MODES:
        return f"Error: mode must be one of {_IMAGE_MODES}, got '{mode}'"
    ok, result = _resolve_model(f"local_image.{mode}", model)
    if not ok:
        return f"Error: {result}"
    try:
        return _generate_image(result, prompt, width, height, steps, output_dir)
    except Exception as exc:
        return f"Error: {exc}"


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
