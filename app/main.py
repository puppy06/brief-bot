"""
TurboLearn-style dashboard: upload a news PDF, poll job status, download MP4.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_OLLAMA_MODEL = os.environ.get("BRIEF_BOT_OLLAMA_MODEL", "llama3.2")

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.templating import Jinja2Templates

from pipeline import create_job, get_job, run_job_async

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(
    title="Brief Bot",
    description="PDF → Ollama summary → ComfyUI + Edge TTS → FFmpeg",
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request) -> HTMLResponse:
    return TEMPLATES.TemplateResponse(
        request,
        "dashboard.html",
        {},
    )


@app.post("/api/jobs")
async def start_job(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_OLLAMA_MODEL),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")
    data = await file.read()
    if len(data) > 40 * 1024 * 1024:
        raise HTTPException(400, "PDF must be 40MB or smaller.")
    job = create_job()
    run_job_async(job.id, data, file.filename, model=model)
    return {"job_id": job.id, "status": job.status}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "id": job.id,
        "status": job.status,
        "step": job.step,
        "progress": job.progress,
        "error": job.error,
        "video_ready": job.status == "done" and job.video_path,
        "brief_ready": job.status == "done" and job.brief_path,
        "meta": job.meta,
        "logs": job.logs[-30:],
    }


@app.get("/api/jobs/{job_id}/video")
async def download_video(job_id: str):
    job = get_job(job_id)
    if not job or job.status != "done" or not job.video_path:
        raise HTTPException(404, "Video not ready")
    p = Path(job.video_path)
    if not p.is_file():
        raise HTTPException(404, "Video file missing")
    return FileResponse(
        p,
        media_type="video/mp4",
        filename=f"briefing_{job_id}.mp4",
    )


@app.get("/api/jobs/{job_id}/brief")
async def download_brief(job_id: str):
    job = get_job(job_id)
    if not job or job.status != "done" or not job.brief_path:
        raise HTTPException(404, "Brief not ready")
    p = Path(job.brief_path)
    if not p.is_file():
        raise HTTPException(404, "Brief file missing")
    return FileResponse(
        p,
        media_type="application/json",
        filename=f"brief_{job_id}.json",
    )
