"""
Content: extract PDF text, ask Ollama for a plain-language summary, then feed video generation.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TypedDict

import ollama
from pypdf import PdfReader


class WorkflowSection(TypedDict):
    script_text: str
    visual_subject: str
    image_path: str


class SimpleBriefPayload(TypedDict):
    """Single summary string — no JSON workflow, headings, or key-point schema."""

    summary: str


OLLAMA_SUMMARY_SYSTEM = """You summarize articles for a general audience.

Rules:
- Write a clear, factual summary of the article the user provides.
- Use plain prose only: one or more paragraphs.
- Do NOT use markdown headings (no # lines). Do NOT label sections (no "Introduction", "Key points", "Summary").
- Do NOT output JSON, XML, or bullet lists unless the source truly requires a short list.
- Do not add a preamble like "Here is a summary" unless you keep it to one short sentence.
"""


def _pdf_to_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n\n".join(parts).strip()


def _compact_source_text(text: str) -> str:
    if not text:
        return text
    t = re.sub(r"[ \t\r\f\v]+", " ", text)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _max_source_chars() -> int:
    try:
        return max(4_000, int(os.environ.get("BRIEF_BOT_MAX_SOURCE_CHARS", "120000")))
    except ValueError:
        return 120_000


def _strip_optional_fence(raw: str) -> str:
    text = raw.strip()
    m = re.match(r"^```(?:\w*)\s*([\s\S]*?)\s*```$", text)
    if m:
        return m.group(1).strip()
    return text


def summarize_article_text(
    source_text: str,
    *,
    model: str = "llama3.2",
    ollama_host: str | None = None,
) -> str:
    """One Ollama call: return plain-text summary only."""
    if ollama_host:
        client = ollama.Client(host=ollama_host)
    else:
        client = ollama.Client()

    body = _compact_source_text(source_text)
    body = body[: _max_source_chars()]

    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": OLLAMA_SUMMARY_SYSTEM},
                {"role": "user", "content": f"Article text:\n\n{body}"},
            ],
            options={"temperature": 0.35, "num_predict": 4096},
        )
    except ollama.ResponseError as e:
        if e.status_code == 404:
            base = model.split(":", 1)[0]
            raise RuntimeError(
                f"Ollama has no local model {model!r} (404). Pull it first: ollama pull {base}"
            ) from e
        raise

    msg = getattr(response, "message", None)
    if msg is None and isinstance(response, dict):
        msg = response.get("message")
    content = getattr(msg, "content", None) if msg is not None else None
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    content = _strip_optional_fence((content or "").strip())
    if not content:
        raise RuntimeError(
            "Ollama returned an empty summary. Is `ollama serve` running, and does "
            f"`ollama list` include the model {model!r}?"
        )
    # Drop leading markdown-style heading lines the model might still emit
    lines = [ln for ln in content.splitlines() if not re.match(r"^#{1,6}\s+", ln.strip())]
    content = "\n".join(lines).strip()
    if len(content) < 40:
        content = (
            content
            + " The article discusses the topics and details found in the source text provided above."
        )
    return content


def build_simple_brief_from_pdf(
    pdf_path: str | Path,
    work_dir: str | Path,
    *,
    model: str = "llama3.2",
    ollama_host: str | None = None,
) -> SimpleBriefPayload:
    path = Path(pdf_path)
    text = _pdf_to_text(path)
    if not text:
        raise ValueError(f"No extractable text in PDF: {path}")
    summary = summarize_article_text(text, model=model, ollama_host=ollama_host)
    return {"summary": summary}


def _guess_visual_subject(summary: str) -> str:
    """Short phrase for ComfyUI from the opening of the summary."""
    s = summary.strip()
    if not s:
        return "the subject matter described in the article"
    chunk = re.split(r"(?<=[.!?])\s+", s, maxsplit=1)[0].strip()
    chunk = re.sub(r"^#+\s*", "", chunk)
    chunk = chunk[:300].strip()
    if len(chunk) < 15:
        return "the main subject described in the article"
    return chunk


def sections_from_simple_brief(payload: SimpleBriefPayload) -> list[WorkflowSection]:
    """Single segment: full summary as narration."""
    summary = (payload.get("summary") or "").strip()
    if not summary:
        raise ValueError("Summary is empty")
    vs = _guess_visual_subject(summary)
    return [{"script_text": summary, "visual_subject": vs, "image_path": ""}]


def simple_brief_to_json_file(payload: SimpleBriefPayload, out_path: str | Path) -> Path:
    p = Path(out_path)
    p.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    return p
