"""
Audio-visual generation: Edge TTS narration + remote ComfyUI (Juggernaut XL).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

import httpx
from dotenv import load_dotenv

from video_editor import image_to_mp4, probe_duration_seconds

load_dotenv(Path(__file__).resolve().parent / ".env")

COMFY_TIMEOUT_SEC = 600


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip().strip('"').strip("'")


def _comfyui_url() -> str:
    url = _env("COMFYUI_URL", "")
    if not url:
        raise RuntimeError(
            "COMFYUI_URL is not set. Point it to your remote ComfyUI backend "
            "(for example: https://<your-colab-host>)."
        )
    return url.rstrip("/")


def _is_vertical_from_image(image_path: Path | None) -> bool:
    if image_path is None or not image_path.is_file():
        return False
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            w, h = im.size
        return h > w
    except Exception:
        return False


def _build_juggernaut_workflow(prompt: str, image_path: Path | None) -> dict[str, Any]:
    """
    Build a ComfyUI API workflow graph for SDXL/Juggernaut.

    KSampler defaults:
    - steps: 35 (within requested 30-40)
    - sampler_name: euler_ancestral ("Euler a")
    - resolution: 1024x1024 or 768x1280 (vertical)
    """
    width, height = (768, 1280) if _is_vertical_from_image(image_path) else (1024, 1024)
    ckpt_name = _env("COMFYUI_CKPT_NAME", "juggernautXL_v8Rundiffusion.safetensors")
    negative = _env(
        "COMFYUI_NEGATIVE_PROMPT",
        "blurry, low quality, artifacts, distorted faces, extra limbs, text watermark",
    )

    # Optional custom workflow override from exported ComfyUI API JSON.
    wf_path = _env("COMFYUI_WORKFLOW_JSON", "")
    if wf_path:
        p = Path(wf_path)
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            for node in data.values():
                if not isinstance(node, dict):
                    continue
                c = node.get("class_type")
                inp = node.get("inputs")
                if not isinstance(inp, dict):
                    continue
                if c == "CLIPTextEncode" and inp.get("text") in ("", "PROMPT", "{prompt}"):
                    inp["text"] = prompt
                elif c == "KSampler":
                    inp["steps"] = 35
                    inp["sampler_name"] = "euler_ancestral"
                elif c == "EmptyLatentImage":
                    inp["width"] = width
                    inp["height"] = height
                elif c == "CheckpointLoaderSimple":
                    inp["ckpt_name"] = ckpt_name
            return data

    # Built-in fallback SDXL/Juggernaut graph.
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(uuid.uuid4().int % 2_147_483_647),
                "steps": 35,
                "cfg": 6.5,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "brief_bot"}},
    }


def _comfy_poll_interval_sec() -> float:
    try:
        return max(0.25, float(os.environ.get("COMFYUI_POLL_INTERVAL_SEC", "1.0")))
    except ValueError:
        return 1.0


def _wait_for_history_completion(
    base: str, prompt_id: str, timeout_sec: int, client: httpx.Client
) -> dict[str, Any]:
    deadline = time.monotonic() + max(5, timeout_sec)
    interval = _comfy_poll_interval_sec()
    while time.monotonic() < deadline:
        r = client.get(f"{base}/history/{prompt_id}")
        r.raise_for_status()
        h = r.json()
        if isinstance(h, dict) and prompt_id in h:
            item = h[prompt_id]
            if isinstance(item, dict) and item.get("outputs"):
                return item
        time.sleep(interval)
    raise RuntimeError(f"ComfyUI history timed out for prompt_id={prompt_id}")


def _extract_output_item(history_obj: dict[str, Any]) -> dict[str, str]:
    outputs = history_obj.get("outputs", {})
    for node_data in outputs.values():
        if not isinstance(node_data, dict):
            continue
        for key in ("videos", "gifs", "images"):
            items = node_data.get(key)
            if isinstance(items, list) and items:
                first = items[0]
                if isinstance(first, dict) and first.get("filename"):
                    return {
                        "filename": str(first["filename"]),
                        "subfolder": str(first.get("subfolder", "")),
                        "type": str(first.get("type", "output")),
                    }
    raise RuntimeError("ComfyUI execution completed but no output media found in history.")


def generate_news_visual(prompt: str, image_path: str | Path | None) -> Path:
    """
    Submit Juggernaut XL workflow to ComfyUI and download resulting media.
    """
    base = _comfyui_url()
    client_id = str(uuid.uuid4())
    img = Path(image_path) if image_path else None
    workflow = _build_juggernaut_workflow(prompt, img)

    limits = httpx.Limits(max_keepalive_connections=8, max_connections=16)
    timeout = httpx.Timeout(COMFY_TIMEOUT_SEC + 60.0, connect=60.0)
    with httpx.Client(timeout=timeout, limits=limits) as c:
        resp = c.post(f"{base}/prompt", json={"prompt": workflow, "client_id": client_id})
        resp.raise_for_status()
        data = resp.json()
        prompt_id = str(data.get("prompt_id") or "")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI /prompt missing prompt_id: {data!r}")

        # Poll GET /history/{prompt_id} until outputs exist (single client = warm TLS).
        history_obj = _wait_for_history_completion(base, prompt_id, COMFY_TIMEOUT_SEC, c)

        item = _extract_output_item(history_obj)
        media = c.get(f"{base}/view?{urlencode(item)}")
        media.raise_for_status()
        content = media.content

    out_dir = Path(__file__).resolve().parent / "data" / "comfyui"
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(item["filename"]).suffix or ".png"
    out_path = out_dir / f"{prompt_id}{ext}"
    out_path.write_bytes(content)
    return out_path


def _build_documentary_prompt(visual_subject: str) -> str:
    subject = (visual_subject or "").strip()
    if not subject:
        subject = "the central real-world subject of this section"
    # Pivoted from 'Format-centric' to 'Content-centric' prompting to increase visual relevance across diverse input types, satisfying the 'Clarity and Usability' requirement of the briefing system.
    final_prompt = (
        f"A professional high-detail photograph of {subject}, cinematic lighting, "
        "National Geographic style, sharp focus, 8k resolution, photorealistic textures "
        "--no newsroom, --no reporter"
    )
    return final_prompt


async def _edge_tts_save(text: str, out_mp3: Path, voice: str) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text.strip(), voice)
    await communicate.save(str(out_mp3))


def synthesize_narration_edge_tts(
    text: str,
    output_mp3: str | Path,
    *,
    voice: str | None = None,
) -> Path:
    out = Path(output_mp3)
    out.parent.mkdir(parents=True, exist_ok=True)
    v = voice or os.environ.get("BRIEF_BOT_EDGE_TTS_VOICE", "en-US-GuyNeural")
    asyncio.run(_edge_tts_save(text, out, v))
    return out


def _edge_tts_concurrency() -> int:
    try:
        return max(1, min(16, int(os.environ.get("BRIEF_BOT_EDGE_TTS_CONCURRENCY", "4"))))
    except ValueError:
        return 4


def synthesize_narrations_edge_tts_parallel(
    items: list[tuple[str, Path]],
    *,
    voice: str | None = None,
) -> None:
    """
    Generate all narration MP3s in one asyncio event loop with bounded parallelism.
    Much faster than sequential ``synthesize_narration_edge_tts`` per segment.
    """
    if not items:
        return
    v = voice or os.environ.get("BRIEF_BOT_EDGE_TTS_VOICE", "en-US-GuyNeural")
    conc = _edge_tts_concurrency()

    async def _run() -> None:
        import edge_tts

        sem = asyncio.Semaphore(conc)

        async def one(text: str, out: Path) -> None:
            out.parent.mkdir(parents=True, exist_ok=True)
            async with sem:
                await edge_tts.Communicate(text.strip(), v).save(str(out))

        await asyncio.gather(*[one(t, Path(o)) for t, o in items])

    asyncio.run(_run())


def generate_ltx_segment_clip(
    *,
    script_text: str,
    visual_subject: str,
    start_image: Path | None,
    theme_context: str,
    work_dir: Path,
    segment_index: int,
    generate_ambient: bool = True,  # retained for compatibility
    on_log: Callable[[str], None] | None = None,
    narration_mp3: Path | None = None,
) -> tuple[Path, Path]:
    _ = generate_ambient
    seg = segment_index
    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    mp3 = Path(narration_mp3) if narration_mp3 is not None else wd / f"seg_{seg:02d}_narration.mp3"
    out_mp4 = wd / f"seg_{seg:02d}_comfy.mp4"

    if narration_mp3 is None:
        if on_log:
            on_log(f"Segment {seg + 1}: TTS (edge-tts)...")
        synthesize_narration_edge_tts(script_text, mp3)
    elif not mp3.is_file():
        raise FileNotFoundError(f"Prerecorded narration missing: {mp3}")
    # Full narration length (still image is held for the whole VO; no arbitrary 20s cap).
    audio_dur = max(2.0, probe_duration_seconds(mp3))

    prompt = _build_documentary_prompt(visual_subject)
    if on_log:
        on_log(f"Segment {seg + 1}: ComfyUI Juggernaut generation...")
    media = generate_news_visual(prompt, start_image)

    if media.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"}:
        shutil.copy2(media, out_mp4)
        return out_mp4, mp3

    image_to_mp4(media, out_mp4, duration_sec=audio_dur)
    return out_mp4, mp3
