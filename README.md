# Brief Bot — Automated Content Briefing System

Ingest a **news PDF**, ask **Ollama** (**Llama 3.2**) for a **plain-text summary** of the article (no JSON workflow or key-point schema), then generate **one visual** with **ComfyUI + Juggernaut XL (SDXL)** and **Edge TTS** narration, and assemble with **FFmpeg** (optional news bed).

The **web dashboard** (`app/`) is a single-page studio: upload a PDF, watch progress, download the MP4 and `brief.json`.

## Merge strategy (audio + video)

**ComfyUI + Juggernaut XL** is responsible for visual generation and stylistic continuity. It is not a speech model: the **news script** is spoken by **Microsoft Edge TTS** (`edge-tts`), which is free and runs locally without an API key.

In the final segment files, narration timing follows the **TTS duration**. If ComfyUI returns still images, the pipeline converts them into timed clips before final assembly.

## Architecture

| Module | Role |
|--------|------|
| `app/main.py` | **FastAPI** app: dashboard UI, upload API, job status, downloads |
| `pipeline.py` | PDF → `processor` → `generator` → `assembler` → optional news bed |
| `processor.py` | PDF text → Ollama → **plain summary**; one section for ComfyUI `visual_subject` |
| `generator.py` | **Edge TTS** → MP3; **ComfyUI API** (`/prompt`, poll `/history`, `/view`) |
| `assembler.py` | Trim/mux segments and **stitch_with_crossfade** |
| `news_processor.py` | Legacy pillar/segment schema (still importable) |
| `multimedia_engine.py` | Slides, lower thirds; legacy helpers |
| `video_editor.py` | **FFmpeg** utilities, xfade/acrossfade, news bed |

`video_editor.py` documents why **FFmpeg** is preferred over a paid video API for assembly (cost, control, portability).

## Prerequisites

- **Python** 3.10+
- **[Ollama](https://ollama.com/)** with a chat model pulled (e.g. `ollama pull llama3.2`)
- **FFmpeg** and **ffprobe** on your `PATH`
- **ComfyUI** reachable from this machine (local or remote, e.g. Colab tunnel URL)

## Install

```bash
cd brief-bot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

| Variable | Purpose |
|----------|---------|
| `COMFYUI_URL` | ComfyUI base URL (example: `https://<ngrok-id>.ngrok-free.app`) |
| `COMFYUI_CKPT_NAME` | Optional checkpoint name (default `juggernautXL_v8Rundiffusion.safetensors`) |
| `COMFYUI_WORKFLOW_JSON` | Optional path to exported ComfyUI API workflow JSON |
| `BRIEF_BOT_OLLAMA_MODEL` | Default Ollama model name in the dashboard (e.g. `llama3.2`) |
| `BRIEF_BOT_MAX_SOURCE_CHARS` | Max PDF text sent to Ollama after whitespace compaction (default `120000`; lower = faster structuring) |
| `BRIEF_BOT_EDGE_TTS_VOICE` | Optional Edge TTS voice (default `en-US-GuyNeural`) |
| `BRIEF_BOT_EDGE_TTS_CONCURRENCY` | Parallel Edge TTS jobs when batching segments (default `4`, max `16`) |
| `COMFYUI_POLL_INTERVAL_SEC` | Seconds between ComfyUI `/history` polls (default `1.0`; `0.25` is slightly snappier) |
| `BRIEF_BOT_NEWS_BED` | Optional path to a news-bed MP3 mixed under the final output |


### ComfyUI on Google Colab (notebook)

The step-by-step Colab notebook lives at **`docs/Brief_Bot_ComfyUI_on_Google_Colab.ipynb`**. Open it in **Google Colab**, enable a **GPU** runtime (**Runtime → Change runtime type → Hardware accelerator → GPU**), then run the cells in order. The notebook installs ComfyUI, pulls the **Juggernaut XL** checkpoint Brief Bot expects, starts the API server, and exposes it with a **Cloudflare quick tunnel** so your machine can reach `COMFYUI_URL` over HTTPS.

**What each part of the notebook does**

| Step | What to run | Purpose |
|------|-------------|---------|
| 1 | Clone ComfyUI under `/content`, `pip install -r requirements.txt`, `wget` the checkpoint into `models/checkpoints/` | Fresh ComfyUI + **Juggernaut XL v8** (`juggernautXL_v8Rundiffusion.safetensors`), matching `COMFYUI_CKPT_NAME` defaults |
| 2 | (Optional) Print `torch` version, `cuda` availability, `device` name | Confirms GPU before long jobs |
| 3 | If restarting: stop the previous ComfyUI process (e.g. `SIGTERM` on `server.pid`) | Avoids “port already in use” when re-running cells |
| 4 | `python main.py --listen 0.0.0.0 --port 8188` (often with log file) | Binds ComfyUI on **all interfaces** so tunnels can forward; **8188** is required for the tunnel step |
| 5 | Loop: `requests.get("http://127.0.0.1:8188", timeout=2)` until `200` or `404` | Waits until ComfyUI is accepting HTTP before tunneling |
| 6 | Download `cloudflared`, run `cloudflared tunnel --url http://127.0.0.1:8188`, parse log for `https://*.trycloudflare.com` | Public **HTTPS** URL; put that value in **`COMFYUI_URL`** in your local `.env` (no trailing slash issues—trim if needed) |

**After the notebook prints `COMFYUI_URL`**

1. Copy the `https://…trycloudflare.com` URL into **`.env`** on the PC running Brief Bot: `COMFYUI_URL=https://…`
2. Keep the **Colab session** (and tunnel process) **alive** while you generate videos; disconnecting ends the GPU and URL.
3. **Alternative tunnels:** ngrok or another HTTPS reverse proxy to port `8188` works the same way; Cloudflare is used in the notebook because it avoids ngrok account setup.

For a deeper explanation of how this fits the overall system, see **`docs/system-overview.md`**.

## Web dashboard

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

Windows single-terminal tip (start Ollama detached first):

```powershell
Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList "serve" -WindowStyle Hidden
```

Open **http://127.0.0.1:8010** — upload a PDF.

Uploads and outputs live under `data/uploads` and `data/outputs`. Jobs are **in memory** (restart clears history).

## `brief.json` shape

After a run, `brief.json` contains the Ollama summary only:

```json
{
  "summary": "Plain prose summary of the article. No headings or key-point labels."
}
```

`generator.py` turns the summary into speech and builds a ComfyUI prompt from the opening of the summary for the still image.

## Usage sketch

```python
from pathlib import Path
from pipeline import run_pdf_to_video_job

final_mp4, payload, meta = run_pdf_to_video_job(
    Path("news.pdf"),
    Path("out/job_1"),
    model="llama3.2",
)
# payload == {"summary": "..."}
```

## License

Not specified; add one if you publish the repo.

