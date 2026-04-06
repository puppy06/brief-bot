"""
Assemble the briefing video with FFmpeg: overlays, cross-fades, and news bed.

Engineering note — why FFmpeg here instead of a paid video API:
- Cost: FFmpeg is free and runs locally or on any VM; per-minute SaaS transcoding
  or timeline APIs accrue fast for iterative newsroom experiments.
- Control: filter graphs give frame-accurate overlays, xfade transitions, and
  loudness tweaks without being limited to template presets.
- Portability: the same commands work in CI, on a laptop, or on a batch worker;
  vendor APIs differ, rate-limit, and lock you into their codec stacks.
Tradeoff: you own the complexity (filter syntax, audio drift). For a hackathon
or internal tool, that is usually cheaper than API dollars and template rigidity.
A paid API still makes sense when you need managed assets, collaboration, or
non-technical editors — not for deterministic assembly from generated inputs.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence


def _run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{proc.stderr}")


def image_to_mp4(
    image_path: str | Path,
    output_path: str | Path,
    duration_sec: float,
    *,
    audio_path: str | Path | None = None,
    fps: int = 30,
) -> Path:
    """
    Encode a still image as H.264 MP4 for ``duration_sec`` seconds.

    If ``audio_path`` is set, mux that audio and let the shorter stream end the clip
    (typical for voiceover that should set the real duration).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = Path(image_path)
    dur = max(0.5, float(duration_sec))

    if audio_path is not None:
        _run_ffmpeg(
            [
                "-loop",
                "1",
                "-i",
                str(img),
                "-i",
                str(audio_path),
                "-c:v",
                "libx264",
                "-tune",
                "stillimage",
                "-pix_fmt",
                "yuv420p",
                "-r",
                str(fps),
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                str(out),
            ]
        )
        return out

    _run_ffmpeg(
        [
            "-loop",
            "1",
            "-i",
            str(img),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=mono:sample_rate=48000",
            "-c:v",
            "libx264",
            "-tune",
            "stillimage",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-t",
            f"{dur:.3f}",
            str(out),
        ]
    )
    return out


def probe_duration_seconds(video_path: str | Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr}")
    return float(proc.stdout.strip())


def probe_video_dimensions(video_path: str | Path) -> tuple[int, int]:
    """Return (width, height) of the first video stream."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr}")
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream in {video_path!r}")
    w = int(streams[0]["width"])
    h = int(streams[0]["height"])
    return w, h


def _even_dim(n: int) -> int:
    """Even dimensions for yuv420p / encoders."""
    return max(2, (max(1, n) + 1) // 2 * 2)


def has_audio_stream(video_path: str | Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return False
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    return len(streams) > 0


def overlay_lower_third(
    base_video: str | Path,
    lower_third_png: str | Path,
    output_video: str | Path,
    *,
    margin_bottom: int = 48,
) -> Path:
    """
    Overlay a pre-rendered lower third (Pillow PNG with alpha) on the full frame.
    """
    out = Path(output_video)
    out.parent.mkdir(parents=True, exist_ok=True)
    filt = (
        f"[1:v]scale=iw:ih,format=yuva420p[lg];"
        f"[0:v][lg]overlay=(W-w)/2:H-h-{margin_bottom}:format=auto[vout]"
    )
    args: list[str] = [
        "-i",
        str(base_video),
        "-loop",
        "1",
        "-i",
        str(lower_third_png),
        "-filter_complex",
        filt,
        "-map",
        "[vout]",
        "-shortest",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
    ]
    if has_audio_stream(base_video):
        args += ["-map", "0:a", "-c:a", "copy"]
    else:
        args += ["-an"]
    args.append(str(out))
    _run_ffmpeg(args)
    return out


def stitch_with_crossfade(
    segment_paths: Sequence[str | Path],
    output_video: str | Path,
    *,
    transition: str = "fade",
    transition_duration: float = 0.8,
    vcodec: str = "libx264",
    crf: int = 20,
    include_audio: bool = True,
    acodec: str = "aac",
    audio_bitrate: str = "192k",
) -> Path:
    """
    Stitch ordered clips (Intro -> Points -> Summary) with xfade transitions.

    Video is scaled and padded to a **common canvas** (max width/height across
    segments, even dimensions) so mixed resolutions (e.g. 1022² vs 1024² from
    ComfyUI) do not break ``xfade``. Audio is resampled to 48 kHz before
    ``acrossfade`` so mixed sample rates (e.g. 24 kHz vs 48 kHz) stay valid.
    """
    paths = [Path(p) for p in segment_paths]
    if len(paths) == 0:
        raise ValueError("Need at least one segment")
    out = Path(output_video)
    out.parent.mkdir(parents=True, exist_ok=True)

    if len(paths) == 1:
        args = ["-i", str(paths[0]), "-c", "copy", str(out)]
        _run_ffmpeg(args)
        return out

    durations = [probe_duration_seconds(p) for p in paths]
    td = transition_duration
    if any(d <= td for d in durations):
        raise ValueError(f"Each segment must be longer than transition duration ({td}s)")

    sizes = [probe_video_dimensions(p) for p in paths]
    W = _even_dim(max(s[0] for s in sizes))
    H = _even_dim(max(s[1] for s in sizes))

    offset = durations[0] - td
    parts: list[str] = []
    inputs: list[str] = []
    for p in paths:
        inputs.extend(["-i", str(p)])

    # Normalize every frame to WxH so xfade inputs match (SDXL / Ken Burns can differ by 2px).
    for i in range(len(paths)):
        parts.append(
            f"[{i}:v]setpts=PTS-STARTPTS,"
            f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[nv{i}]"
        )

    cur_v = "nv0"
    for i in range(1, len(paths)):
        out_v = f"vx{i}"
        parts.append(
            f"[{cur_v}][nv{i}]xfade=transition={transition}:duration={td}:offset={offset:.3f}[{out_v}]"
        )
        cur_v = out_v
        if i < len(paths) - 1:
            offset += durations[i] - td

    maps = ["-map", f"[{cur_v}]"]
    filter_complex = ";".join(parts)

    if include_audio:
        if not all(has_audio_stream(p) for p in paths):
            raise ValueError("include_audio=True requires an audio stream in every segment")
        parts.append("[0:a]asetpts=PTS-STARTPTS,aresample=48000[a0]")
        cur_a = "a0"
        for i in range(1, len(paths)):
            ai = f"ar{i}"
            parts.append(f"[{i}:a]asetpts=PTS-STARTPTS,aresample=48000[{ai}]")
            out_a = f"a{i}"
            parts.append(
                f"[{cur_a}][{ai}]acrossfade=d={td}:c1=tri:c2=tri[{out_a}]"
            )
            cur_a = out_a
        filter_complex = ";".join(parts)
        maps += ["-map", f"[{cur_a}]"]

    args = (
        inputs
        + [
            "-filter_complex",
            filter_complex,
            *maps,
            "-c:v",
            vcodec,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if include_audio:
        args += ["-c:a", acodec, "-b:a", audio_bitrate]
    else:
        args += ["-an"]
    args.append(str(out))
    _run_ffmpeg(args)
    return out


def mux_news_bed(
    main_video: str | Path,
    bed_audio: str | Path,
    output_video: str | Path,
    *,
    bed_volume: float = 0.12,
    voice_volume: float = 1.0,
) -> Path:
    """
    Mix anchor/dialogue (from main video's audio) with a low 'news bed' music track.

    bed_volume should stay low so narration remains intelligible (typically 0.08–0.18).
    """
    out = Path(output_video)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not has_audio_stream(main_video):
        raise ValueError("main_video must contain an audio track to mix with the news bed")

    filt = (
        f"[0:a]volume={voice_volume}[a0];"
        f"[1:a]volume={bed_volume}[a1];"
        f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=2[aout]"
    )
    _run_ffmpeg(
        [
            "-i",
            str(main_video),
            "-stream_loop",
            "-1",
            "-i",
            str(bed_audio),
            "-filter_complex",
            filt,
            "-map",
            "0:v",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-shortest",
            str(out),
        ]
    )
    return out


def assemble_briefing_video(
    stitched_video: str | Path,
    lower_third_png: str | Path,
    bed_audio: str | Path,
    output_video: str | Path,
) -> Path:
    """
    Convenience: overlay lower third then mix news bed in one pass when possible.

    For timed lower thirds per segment, call overlay_lower_third per clip before
    stitch_with_crossfade, or use enable='between(t,start,end)' in a custom graph.
    """
    tmp = Path(stitched_video).with_suffix(".lt.mp4")
    overlay_lower_third(stitched_video, lower_third_png, tmp)
    return mux_news_bed(tmp, bed_audio, output_video)
