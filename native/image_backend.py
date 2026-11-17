"""Apple-native image generation (macOS 27+, Core AI): the Python side of ``ollamamcp-imagegen``.

This is OllamaMCP's optional, platform-specific layer. The cross-platform core (server.py) imports it lazily
and registers ``local_image`` only when ``platform_supported()`` and ``helper_path()`` say it can run here, so
on Linux, Windows or older macOS the server starts exactly as before, just without the tool. Keep everything
Apple-specific inside ``native/`` so it can later move out in one piece into a separate Apple-native MCP.

Models are Core AI diffusion bundles (a folder with ``metadata.json`` + ``*.aimodel``), discovered on disk under
``native_models_dir``. Their ids are backend-qualified (``coreai/flux2-klein-4b``) so they can sit in
``tool_defaults`` beside Ollama model names without colliding.
"""
from __future__ import annotations

import asyncio
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

BACKEND = "coreai"
HELPER_NAME = "ollamamcp-imagegen"
NATIVE_DIR = Path(__file__).parent
DEFAULT_HELPER = NATIVE_DIR / ".build" / "release" / HELPER_NAME
DEFAULT_MODELS_DIR = Path.home() / "Library" / "Application Support" / "OllamaMCP" / "models" / BACKEND

# Diffusion families the helper's FlowTransformerPipeline runs (CoreAIDiffusionPipeline DiffusionConfig raw
# values). Anything else, e.g. a Z-Image port that needs Python + torch, is skipped at discovery.
SUPPORTED_TYPES = {"flux2", "sana-sprint"}

# Measured helper peaks on an M1 Max (Looking Glass, 2026-09-19 / 10-13, FLUX.2 [klein] 4B int4): a new image
# 6.8 GB, an edit on the full reference grid 7.25 GB. Used for the memory headroom check, plus a margin.
PEAK_GB = {"generate": 6.8, "edit": 7.3}
MARGIN_GB = 1.5
GENERATE_TIMEOUT_S = 600   # the first run under a new process name specializes the model (~20–120 s)


class NativeImageError(Exception):
    """A failure to report to the caller as-is."""


@dataclass(frozen=True)
class NativeModel:
    id: str            # "coreai/flux2-klein-4b"
    name: str          # bundle's own name, "FLUX.2-klein-4B"
    path: Path
    family: str        # "flux2" | "sana-sprint"
    can_edit: bool     # has a VAE encoder, so img2img (edits) works
    size_gb: float


# ── Availability ──────────────────────────────────────────────────────────────
def platform_supported() -> tuple[bool, str]:
    """Core AI needs macOS 27 on Apple silicon."""
    if sys.platform != "darwin":
        return False, "Core AI image generation is macOS-only"
    release = platform.mac_ver()[0]
    try:
        major = int(release.split(".")[0])
    except ValueError:
        return False, f"couldn't read the macOS version ({release!r})"
    if major < 27:
        return False, f"Core AI image generation needs macOS 27 (this is {release})"
    if platform.machine() != "arm64":
        return False, "Core AI image generation needs Apple silicon"
    return True, ""


def helper_path(configured: Optional[str] = None) -> Optional[Path]:
    """The built helper: config.json `native_image_helper`, else native/.build/release/."""
    p = Path(configured).expanduser() if configured else DEFAULT_HELPER
    return p if p.is_file() else None


def models_dir(configured: Optional[str] = None) -> Path:
    return Path(configured).expanduser() if configured else DEFAULT_MODELS_DIR


# ── Discovery ─────────────────────────────────────────────────────────────────
def model_id(bundle_name: str) -> str:
    """'FLUX.2-klein-4B' -> 'coreai/flux2-klein-4b'."""
    slug = re.sub(r"[^a-z0-9]+", "-", bundle_name.lower().replace(".", "")).strip("-")
    return f"{BACKEND}/{slug}"


def _dir_size_gb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / 1e9, 1)


def discover(root: Path) -> dict[str, NativeModel]:
    """Every runnable Core AI diffusion bundle directly under `root`, keyed by id."""
    found: dict[str, NativeModel] = {}
    if not root.is_dir():
        return found
    for bundle in sorted(p for p in root.iterdir() if p.is_dir()):
        meta_path = bundle / "metadata.json"
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            continue
        family = (meta.get("diffusion") or {}).get("type")
        if meta.get("kind") != "diffusion" or family not in SUPPORTED_TYPES:
            continue
        assets = meta.get("assets") or {}
        name = meta.get("name") or bundle.name
        m = NativeModel(
            id=model_id(name),
            name=name,
            path=bundle,
            family=family,
            can_edit="vae_encoder" in assets,
            size_gb=_dir_size_gb(bundle),
        )
        found[m.id] = m
    return found


# ── Memory ────────────────────────────────────────────────────────────────────
def available_ram_gb() -> Optional[float]:
    """Memory macOS can hand out without swapping: free + inactive + speculative + purgeable pages."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    page = re.search(r"page size of (\d+) bytes", out)
    page_size = int(page.group(1)) if page else 16384
    pages = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        m = re.search(rf"{key}:\s+(\d+)", out)
        if m:
            pages += int(m.group(1))
    return round(pages * page_size / 1e9, 1) if pages else None


def needed_gb(editing: bool) -> float:
    return PEAK_GB["edit" if editing else "generate"] + MARGIN_GB


# ── Running the helper ────────────────────────────────────────────────────────
async def generate(
    helper: Path,
    model: NativeModel,
    prompt: str,
    output: Path,
    seed: int,
    steps: Optional[int] = None,
    edit_image: Optional[Path] = None,
    on_step: Optional[Callable[[int, int], Awaitable[None]]] = None,
) -> dict:
    """Run one generation (or edit). Returns the helper's `done` event: path, seconds, width, height."""
    if edit_image is not None and not model.can_edit:
        raise NativeImageError(f"{model.id} can't edit images (its bundle has no image encoder)")
    args = [str(helper), "--model", str(model.path), "--prompt", prompt,
            "--seed", str(seed), "--output", str(output)]
    if steps:
        args += ["--steps", str(steps)]
    if edit_image is not None:
        # Full grid keeps chained edits crisp; half (the coreai-models README example) blurs and drifts.
        args += ["--input-image", str(edit_image), "--reference-grid", "full"]
    return await _run(args, expect="done", on_step=on_step, timeout=GENERATE_TIMEOUT_S)


async def clear_cache(helper: Path, model: NativeModel) -> dict:
    """Drop Core AI's specialized copy of a bundle (~its size, never reclaimed on its own). Run before deleting it."""
    return await _run([str(helper), "--clear-cache", "--model", str(model.path)], expect="cleared", timeout=60)


async def _run(args: list[str], expect: str, timeout: float,
               on_step: Optional[Callable[[int, int], Awaitable[None]]] = None) -> dict:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    result: Optional[dict] = None
    error: Optional[str] = None

    async def read_events() -> None:
        nonlocal result, error
        assert proc.stdout is not None
        async for raw in proc.stdout:
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            kind = event.get("event")
            if kind == "step" and on_step is not None:
                try:
                    await on_step(int(event.get("step", 0)), int(event.get("total", 0)))
                except Exception:
                    pass  # progress is best-effort; never fail an image over it
            elif kind == "error":
                error = event.get("message") or "unknown error"
            elif kind == expect:
                result = event

    try:
        await asyncio.wait_for(read_events(), timeout=timeout)
        await asyncio.wait_for(proc.wait(), timeout=10)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise NativeImageError(f"{HELPER_NAME} timed out after {int(timeout)} s")
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise

    if error:
        raise NativeImageError(error)
    if result is None:
        stderr = (await proc.stderr.read()).decode(errors="replace").strip() if proc.stderr else ""
        raise NativeImageError(
            f"{HELPER_NAME} exited ({proc.returncode}) without a result"
            + (f": {stderr[-500:]}" if stderr else "")
        )
    return result
