"""
End-to-end job: PDF → Ollama plain-text summary → Edge TTS + ComfyUI visual →
assemble → optional news bed.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from assembler import assemble_program, finalize_segment_for_program
from generator import generate_ltx_segment_clip, synthesize_narrations_edge_tts_parallel
from processor import (
    SimpleBriefPayload,
    build_simple_brief_from_pdf,
    sections_from_simple_brief,
    simple_brief_to_json_file,
)
from video_editor import mux_news_bed, probe_duration_seconds


def run_pdf_to_video_job(
    pdf_path: Path,
    work_dir: Path,
    *,
    model: str = "llama3.2",
    ollama_host: str | None = None,
    news_bed_path: Path | None = None,
    on_log: Callable[[str], None] | None = None,
) -> tuple[Path, SimpleBriefPayload, dict[str, Any]]:
    """
    Run the pipeline: PDF text → Ollama summary → one narration + one visual → optional bed.

    Returns ``(final_video_path, brief_payload, meta)``.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if on_log:
        on_log("Summarizing article with Ollama…")
    payload = build_simple_brief_from_pdf(
        pdf_path, work_dir, model=model, ollama_host=ollama_host
    )
    simple_brief_to_json_file(payload, work_dir / "brief.json")

    sections = sections_from_simple_brief(payload)
    theme = (payload["summary"] or "")[:2000]
    img_root = work_dir / "extracted_images"
    seg_dir = work_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    mp3_paths = [seg_dir / f"seg_{i:02d}_narration.mp3" for i in range(len(sections))]
    if on_log:
        on_log("Synthesizing narration (edge-tts, parallel)…")
    synthesize_narrations_edge_tts_parallel(
        [(sec["script_text"], mp3_paths[i]) for i, sec in enumerate(sections)]
    )

    segment_finals: list[Path] = []
    for i, sec in enumerate(sections):
        raw = (sec.get("image_path") or "").strip()
        start = (img_root / raw) if raw else None
        if start is not None and not start.is_file():
            start = None
        if on_log:
            on_log(f"LTX segment {i + 1}/{len(sections)}…")
        raw_mp4, narr_mp3 = generate_ltx_segment_clip(
            script_text=sec["script_text"],
            visual_subject=sec["visual_subject"],
            start_image=start,
            theme_context=theme,
            work_dir=seg_dir,
            segment_index=i,
            generate_ambient=True,
            on_log=on_log,
            narration_mp3=mp3_paths[i],
        )
        final_mp4 = seg_dir / f"seg_{i:02d}_final.mp4"
        finalize_segment_for_program(
            raw_mp4,
            narr_mp3,
            final_mp4,
            mix_ambient=(i != 0),
            title_ken_burns=(i == 0),
        )
        segment_finals.append(final_mp4)

    stitched = work_dir / "stitched.mp4"
    if on_log:
        on_log("Assembling final video…")
    td = 0.8
    if len(segment_finals) > 1:
        for p in segment_finals:
            if probe_duration_seconds(p) <= td:
                raise RuntimeError(
                    f"Segment {p} is shorter than cross-fade duration ({td}s)."
                )
    assemble_program(segment_finals, stitched, transition_duration=td)

    final_path = work_dir / "briefing_final.mp4"
    bed = news_bed_path
    if bed is None:
        env_bed = os.environ.get("BRIEF_BOT_NEWS_BED")
        if env_bed:
            bed = Path(env_bed)
    if bed and bed.is_file():
        if on_log:
            on_log("Mixing news bed…")
        mux_news_bed(stitched, bed, final_path, bed_volume=0.12)
    else:
        shutil.copy2(stitched, final_path)

    meta = {
        "workflow": "ollama_comfy_edge",
        "tts": "edge-tts",
        "video_model": "ComfyUI Juggernaut XL",
        "news_bed": str(bed) if bed and bed.is_file() else None,
    }
    return final_path, payload, meta


@dataclass
class JobState:
    id: str
    status: str = "queued"
    step: str = ""
    progress: float = 0.0
    error: str | None = None
    video_path: str | None = None
    brief_path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    logs: list[str] = field(default_factory=list)


_jobs: dict[str, JobState] = {}
_jobs_lock = threading.Lock()
DATA_DIR = Path(__file__).resolve().parent / "data"
UPLOADS = DATA_DIR / "uploads"
OUTPUTS = DATA_DIR / "outputs"


def _ensure_dirs() -> None:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)


def create_job() -> JobState:
    _ensure_dirs()
    jid = uuid.uuid4().hex[:12]
    job = JobState(id=jid)
    with _jobs_lock:
        _jobs[jid] = job
    return job


def get_job(job_id: str) -> JobState | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def _append_log(job: JobState, msg: str) -> None:
    job.logs.append(msg)
    if len(job.logs) > 200:
        job.logs = job.logs[-200:]


def run_job_async(
    job_id: str,
    pdf_bytes: bytes,
    filename: str,
    *,
    model: str = "llama3.2",
) -> None:
    def worker() -> None:
        job = get_job(job_id)
        if not job:
            return
        _ensure_dirs()
        safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ")[:180]
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"
        pdf_path = UPLOADS / f"{job_id}_{safe_name}"
        work_dir = OUTPUTS / job_id

        try:
            job.status = "running"
            job.step = "Saving upload"
            job.progress = 0.05
            pdf_path.write_bytes(pdf_bytes)
            work_dir.mkdir(parents=True, exist_ok=True)

            def on_log(msg: str) -> None:
                job.step = msg
                _append_log(job, msg)
                job.progress = min(0.92, job.progress + 0.09)

            job.progress = 0.1
            final, _payload, meta = run_pdf_to_video_job(
                pdf_path,
                work_dir,
                model=model,
                on_log=on_log,
            )
            job.video_path = str(final)
            job.brief_path = str(work_dir / "brief.json")
            job.meta = meta
            job.status = "done"
            job.step = "Complete"
            job.progress = 1.0
        except Exception as e:
            job.status = "error"
            job.error = str(e)
            job.step = "Failed"
            _append_log(job, str(e))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
