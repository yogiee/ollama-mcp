#!/usr/bin/env python3
"""
ollama-mcp MCP server — exposes Ollama models as tools in Claude Code sessions.
Tool-to-model mappings are read from registry.json at startup.

The core is cross-platform and Ollama-only. `local_image` comes from the optional Apple-native layer in
native/ (Core AI, macOS 27+) and is registered only where that layer can run.
"""
import asyncio
import json
import os
import secrets
import sys
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

import ollama
from mcp.server.fastmcp import Context, FastMCP

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

# ── Apple-native layer (optional) ──────────────────────────────────────────────
# Imported lazily so the core never depends on it. Native model ids are backend-qualified
# ("coreai/flux2-klein-4b") and resolved against bundles on disk instead of Ollama's model list.
_native = None                       # the native.image_backend module, when it can run here
_native_helper: Optional[Path] = None
_native_models: dict = {}            # id → NativeModel
_native_models_dir: Optional[Path] = None
_native_unavailable = "Apple-native layer not loaded"
_NATIVE_PREFIX = "coreai/"

try:
    from native import image_backend as _nb

    _ok, _why = _nb.platform_supported()
    _helper = _nb.helper_path(_config.get("native_image_helper"))
    if not _ok:
        _native_unavailable = _why
    elif _helper is None:
        _native_unavailable = (
            f"{_nb.HELPER_NAME} isn't built. Run: swift build -c release --package-path native"
        )
    else:
        _native, _native_helper = _nb, _helper
        _native_models_dir = _nb.models_dir(_config.get("native_models_dir"))
        _native_models = _nb.discover(_native_models_dir)
except Exception as _exc:  # never let the optional layer break the core
    _native_unavailable = f"Apple-native layer failed to load: {_exc}"


def _is_native(model: str) -> bool:
    return model.startswith(_NATIVE_PREFIX)


# ── Startup: verify reachability + warn on missing defaults ───────────────────
_available_models: list[str] = []
_ollama_reachable = False

try:
    _available_models = [m.model for m in _client.list().models]
    _ollama_reachable = True
except Exception as exc:
    print(f"WARNING: Ollama not reachable at {OLLAMA_HOST}: {exc}", file=sys.stderr)

for _tool, _model in TOOL_DEFAULTS.items():
    if not _model:
        continue
    if _tool.startswith("local_image."):
        if _native is None:
            continue  # local_image isn't registered on this host; nothing to warn about
        if _model not in _native_models:
            print(
                f"WARNING: {_tool} default model '{_model}' has no bundle in {_native_models_dir}",
                file=sys.stderr,
            )
    elif _ollama_reachable and _model not in _available_models:
        print(
            f"WARNING: {_tool} default model '{_model}' not found in Ollama. "
            "Run: python maintenance.py --sync",
            file=sys.stderr,
        )

# ── Helpers ────────────────────────────────────────────────────────────────────
def _resolve_model(tool: str, override: Optional[str]) -> tuple[bool, str]:
    """Return (ok, model_name) on success or (False, error_message) on failure."""
    m = override or TOOL_DEFAULTS.get(tool)
    if not m:
        return False, f"No model configured for {tool}. Run: python maintenance.py --sync"
    if _is_native(m):
        if _native is None:
            return False, f"Model '{m}' needs the Apple-native layer: {_native_unavailable}"
        if m not in _native_models:
            have = ", ".join(sorted(_native_models)) or "none"
            return False, (
                f"Model '{m}' has no Core AI bundle in {_native_models_dir} (installed: {have}). "
                "Set native_models_dir in config.json, or download the bundle there."
            )
        return True, m
    if not _ollama_reachable:
        return False, f"Ollama not reachable at {OLLAMA_HOST}"
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
# only maps mode → model. "photo" → photorealism, "design" → text/UI/illustration/edits.
_IMAGE_MODES = ("photo", "design")


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


def _image_output_path(model: str, prompt: str, output_dir: Optional[str]) -> Path:
    """A fresh, descriptive PNG path in the resolved save dir (created if needed)."""
    out_dir = _image_output_dir(output_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:  # chosen dir unwritable after all → guaranteed-writable fallback
        out_dir = Path.home() / "Pictures" / "Generated_images"
        out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = model.split("/")[-1].replace(":", "-")
    safe = "".join(c if c.isalnum() or c in " _-" else "" for c in prompt[:45]).strip().replace(" ", "_")
    return out_dir / f"{stamp}_{tag}_{safe}.png"


def _make_room_for_image(editing: bool) -> str:
    """Free memory for an image if it won't fit beside what's loaded. Returns a note for the result, or "".

    Unloads only OllamaMCP's own tool-default models (they reload on their next call). Models other
    apps loaded are left alone and named in the note, so the caller can decide; a consumer that knows
    its own models (Looking Glass's voice and chat) should free them before calling local_image.
    """
    need = _native.needed_gb(editing)
    free = _native.available_ram_gb()
    if free is None or free >= need or not _ollama_reachable:
        return ""
    try:
        resident = [m.model for m in _client.ps().models]
    except Exception:
        return ""
    ours = {m for t, m in TOOL_DEFAULTS.items() if m and not t.startswith("local_image.")}
    unloaded = []
    for name in resident:
        if name in ours:
            try:
                _client.generate(model=name, keep_alive=0)  # empty request + keep_alive 0 = unload
                unloaded.append(name)
            except Exception:
                pass
    others = [m for m in resident if m not in unloaded]
    parts = [f"Memory was short ({free:.1f} GB free, ~{need:.1f} GB needed)"]
    if unloaded:
        parts.append(f"unloaded {', '.join(unloaded)}")
    if others:
        parts.append(f"still loaded by other apps: {', '.join(others)} (may swap)")
    return "; ".join(parts) + ". "


# ── MCP server ─────────────────────────────────────────────────────────────────
mcp = FastMCP("ollama-mcp")


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
            # Top-level arg, passed explicitly: thinking models (gemma4) think by default,
            # so think=False is what actually turns it off. Ollama ignores options["think"].
            think=thinking,
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


_image_lock = asyncio.Lock()  # one image at a time: each needs ~7 GB


async def local_image(
    prompt: str,
    mode: str = "photo",
    edit_image: Optional[str] = None,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    model: Optional[str] = None,
    output_dir: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> str:
    """Generate an image on-device from a text prompt, or edit an existing image. Returns where the PNG was saved.

    mode="photo"  -> photorealism, portraits, landscapes, product shots
    mode="design" -> text-in-image, logos, UI mockups, posters, illustration
    edit_image    -> path or http(s) URL of an image to change; the prompt describes the change
                     ("make the sky stormy"). To chain edits, pass the previous result's path.
    seed          -> repeat a result exactly (same prompt + seed + model); random if omitted
    steps         -> diffusion steps; the model's own default is best (Klein: 4)
    Pick mode from the user's request; pass model= to force a specific model.
    Output is 1024x1024. Takes ~20-60 s (the very first run on a machine also prepares the model).
    Saves to ./Generated_images in the current project, else ~/Pictures/Generated_images.
    The reply ends with the saved path.
    """
    if mode not in _IMAGE_MODES:
        return f"Error: mode must be one of {_IMAGE_MODES}, got '{mode}'"
    ok, result = _resolve_model(f"local_image.{mode}", model)
    if not ok:
        return f"Error: {result}"
    if not _is_native(result):
        return f"Error: '{result}' isn't an image model this server can run (expected a {_NATIVE_PREFIX}… model)"
    native_model = _native_models[result]

    temp_input: Optional[Path] = None
    edit_path: Optional[Path] = None
    if edit_image:
        try:
            if edit_image.startswith(("http://", "https://")):
                suffix = Path(edit_image.split("?")[0]).suffix or ".png"
                fd, name = tempfile.mkstemp(suffix=suffix, prefix="ollamamcp-edit-")
                with os.fdopen(fd, "wb") as f:
                    f.write(_load_image(edit_image))
                temp_input = edit_path = Path(name)
            else:
                edit_path = Path(edit_image).expanduser()
                if not edit_path.is_file():
                    return f"Error: edit_image '{edit_image}' not found"
        except Exception as exc:
            return f"Error loading edit_image '{edit_image}': {exc}"

    seed = seed if seed is not None else secrets.randbelow(2**31)

    async def on_step(step: int, total: int) -> None:
        if ctx is not None:
            await ctx.report_progress(step, total, f"step {step}/{total}")

    try:
        async with _image_lock:
            note = await asyncio.to_thread(_make_room_for_image, edit_path is not None)
            out_path = _image_output_path(result, prompt, output_dir)
            done = await _native.generate(
                _native_helper, native_model, prompt, out_path, seed,
                steps=steps, edit_image=edit_path, on_step=on_step,
            )
    except Exception as exc:
        return f"Error: {exc}"
    finally:
        if temp_input is not None:
            temp_input.unlink(missing_ok=True)

    verb = "Edited" if edit_path is not None else "Generated"
    return (
        f"Seed {seed}. {verb} in {float(done.get('seconds', 0)):.1f} s "
        f"({done.get('width')}x{done.get('height')}, {result}). {note}"
        f"Image saved to {done.get('path', out_path)}"
    )


# Registered only where the Apple-native layer can run (macOS 27+, helper built). Elsewhere the
# server starts without it, rather than offering a tool that can only fail.
if _native is not None:
    mcp.tool()(local_image)


@mcp.tool()
def list_local_models() -> list[dict]:
    """List installed models (Ollama, plus on-device Core AI image models where available) with
    capabilities, registry status, and which tool(s) each is the default for."""
    registry_models: dict = _registry.get("models", {})
    default_for: dict[str, list[str]] = {}
    for tool, model in TOOL_DEFAULTS.items():
        if model:
            default_for.setdefault(model, []).append(tool)
    result = []
    if _ollama_reachable:
        for m in _client.list().models:
            name = m.model
            entry = registry_models.get(name, {})
            benchmarked = any(
                isinstance(v, dict) and v.get("score") is not None
                for v in entry.get("benchmarks", {}).values()
            )
            result.append({
                "name": name,
                "backend": "ollama",
                "size_gb": entry.get("size_gb"),
                "capabilities": entry.get("ollama_capabilities", []),
                "in_registry": name in registry_models,
                "benchmarked": benchmarked,
                "tool_default_for": ", ".join(default_for.get(name, [])) or None,
                "avoid_for": entry.get("avoid_for", []),
            })
    else:
        result.append({"error": f"Ollama not reachable at {OLLAMA_HOST}"})
    for nm in _native_models.values():
        result.append({
            "name": nm.id,
            "backend": _native.BACKEND,
            "size_gb": nm.size_gb,
            "capabilities": ["image", "image_edit"] if nm.can_edit else ["image"],
            "in_registry": nm.id in registry_models,
            "benchmarked": False,  # image models are never benchmarked (no objective grade)
            "tool_default_for": ", ".join(default_for.get(nm.id, [])) or None,
            "avoid_for": [],
        })
    return result


if __name__ == "__main__":
    mcp.run()
