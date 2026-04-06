"""
Newsroom workflow: extract structured pillars from source material via Ollama,
then produce a broadcast-style script timeline as JSON-ready segments.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TypedDict

import ollama
from pypdf import PdfReader


class ScriptSegment(TypedDict):
    timestamp_start: float
    script_text: str
    visual_description: str


class NewsBriefPayload(TypedDict):
    headline: str
    lead: str
    key_developments: list[str]
    impact: str
    segments: list[ScriptSegment]


OLLAMA_SYSTEM = """You are a senior TV news producer. Your job is to turn raw input into a tight broadcast package.

Rules:
1. Extract exactly these News Pillars: Headline, Lead, exactly 3 Key Developments, Impact.
2. Write all spoken copy in "Broadcast Voice": short, punchy sentences; present tense where natural; no filler; no markdown; no stage directions inside script lines.
3. The full read should target roughly 3–5 minutes when spoken (~450–750 words total across segments). Pace segments so an anchor can read them comfortably.
4. For each segment, give a concrete visual_description for B-roll or graphics (what the viewer should see).
5. Output ONE valid JSON object only, matching this schema (no code fences, no commentary):
{
  "headline": string,
  "lead": string,
  "key_developments": [string, string, string],
  "impact": string,
  "segments": [
    {
      "timestamp_start": number (seconds from start of final video, monotonic increasing),
      "script_text": string,
      "visual_description": string
    }
  ]
}

Segment order: intro (headline + lead) -> one segment per key development -> closing (impact + sign-off). Assign timestamp_start in seconds (e.g. 0, 25, 55, ...)."""


def _pdf_to_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n\n".join(parts).strip()


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text)
    if m:
        return m.group(1).strip()
    return text


def _parse_brief_json(content: str) -> NewsBriefPayload:
    data: Any = json.loads(_strip_json_fence(content))
    required_top = {"headline", "lead", "key_developments", "impact", "segments"}
    if not isinstance(data, dict) or not required_top.issubset(data.keys()):
        raise ValueError("LLM JSON missing required keys")
    kd = data["key_developments"]
    if not isinstance(kd, list) or len(kd) != 3:
        raise ValueError("key_developments must be a list of exactly 3 strings")
    segs = data["segments"]
    if not isinstance(segs, list) or not segs:
        raise ValueError("segments must be a non-empty list")
    for i, s in enumerate(segs):
        if not isinstance(s, dict):
            raise ValueError(f"segment {i} is not an object")
        for k in ("timestamp_start", "script_text", "visual_description"):
            if k not in s:
                raise ValueError(f"segment {i} missing {k}")
    return data  # type: ignore[return-value]


def build_news_brief_from_text(
    source_text: str,
    *,
    model: str = "llama3.2",
    ollama_host: str | None = None,
) -> NewsBriefPayload:
    """
    Call Ollama to produce pillars + broadcast-voice segments with timestamps.
    """
    if ollama_host:
        client = ollama.Client(host=ollama_host)
    else:
        client = ollama.Client()

    user_msg = (
        "Source material follows. Produce the JSON package.\n\n---\n\n" + source_text[:120_000]
    )

    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": OLLAMA_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            options={"temperature": 0.35},
        )
    except ollama.ResponseError as e:
        if e.status_code == 404:
            base = model.split(":", 1)[0]
            raise RuntimeError(
                f"Ollama has no local model {model!r} (404). "
                f"Pull it first: ollama pull {base}  "
                f"Or run ollama list and type one of those names in the dashboard."
            ) from e
        raise
    msg = getattr(response, "message", None)
    if msg is None and isinstance(response, dict):
        msg = response.get("message")
    content = getattr(msg, "content", None) if msg is not None else None
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    if not content:
        raise RuntimeError("Ollama returned an empty message")
    return _parse_brief_json(content)


def build_news_brief_from_pdf(
    pdf_path: str | Path,
    *,
    model: str = "llama3.2",
    ollama_host: str | None = None,
) -> NewsBriefPayload:
    path = Path(pdf_path)
    text = _pdf_to_text(path)
    if not text:
        raise ValueError(f"No extractable text in PDF: {path}")
    return build_news_brief_from_text(text, model=model, ollama_host=ollama_host)


def brief_to_json_file(brief: NewsBriefPayload, out_path: str | Path) -> Path:
    p = Path(out_path)
    p.write_text(json.dumps(brief, indent=2), encoding="utf-8")
    return p
