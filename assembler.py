"""
FFmpeg assembly: merge LTX-2.3 clips with Edge TTS narration, optional ambient bed,
and stitch segments into one program.

Merge strategy (see README): LTX-2.3 supplies **motion continuity** and **ambient
sound** when ``generate_audio`` is enabled; **Edge TTS** carries the actual news
narration. This module mixes those layers, trims to the narration clock, and
concatenates with cross-fades.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

from video_editor import (
    has_audio_stream,
    probe_duration_seconds,
    stitch_with_crossfade,
)


def _run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{proc.stderr}")


def trim_video_to_duration(
    video_path: str | Path,
    duration_sec: float,
    output_path: str | Path,
) -> Path:
    """Trim video (and bundled audio) to ``duration_sec`` seconds."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    d = max(0.1, float(duration_sec))
    _run_ffmpeg(
        [
            "-i",
            str(video_path),
            "-t",
            f"{d:.3f}",
            "-c",
            "copy",
            str(out),
        ]
    )
    return out


def mux_video_with_narration_only(
    video_path: str | Path,
    narration_mp3: str | Path,
    output_path: str | Path,
) -> Path:
    """Replace or attach narration when the video has no usable audio stream."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-i",
            str(video_path),
            "-i",
            str(narration_mp3),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(out),
        ]
    )
    return out


def combine_narration_with_ltx_ambient(
    ltx_video_path: str | Path,
    narration_mp3: str | Path,
    output_path: str | Path,
    *,
    ambient_gain: float = 0.22,
    voice_gain: float = 1.0,
) -> Path:
    """
    Mix TTS (primary) with LTX's generated ambient track (secondary).

    If ``ltx_video_path`` has no audio, narration is muxed alone.
    """
    vid = Path(ltx_video_path)
    narr = Path(narration_mp3)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not has_audio_stream(vid):
        return mux_video_with_narration_only(vid, narr, out)

    adur = probe_duration_seconds(narr)
    filt = (
        f"[0:a]volume={ambient_gain}[amb];"
        f"[1:a]volume={voice_gain}[vo];"
        f"[amb][vo]amix=inputs=2:duration=shortest:dropout_transition=2[aout]"
    )
    _run_ffmpeg(
        [
            "-i",
            str(vid),
            "-i",
            str(narr),
            "-filter_complex",
            filt,
            "-map",
            "0:v",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(out),
        ]
    )
    return out


def finalize_segment_for_program(
    ltx_video_path: str | Path,
    narration_mp3: str | Path,
    output_path: str | Path,
    *,
    mix_ambient: bool = True,
    title_ken_burns: bool = False,
) -> Path:
    """
    Trim LTX output to narration length (if needed) and merge audio layers.

    Returns a segment MP4 ready for ``stitch_with_crossfade``.
    """
    vid = Path(ltx_video_path)
    narr = Path(narration_mp3)
    adur = probe_duration_seconds(narr)
    vdur = probe_duration_seconds(vid)
    tmp: Path | None = None
    if vdur > adur + 0.05:
        tmp = Path(output_path).with_suffix(".pretrim.mp4")
        trim_video_to_duration(vid, adur, tmp)
        src: Path = tmp
    else:
        src = vid

    if title_ken_burns:
        # Title card should feel like a documentary intro instead of a static hold.
        kb_tmp = Path(output_path).with_suffix(".kenburns.mp4")
        zoom = 1.12
        d = max(0.5, float(adur))
        _run_ffmpeg(
            [
                "-i",
                str(src),
                "-vf",
                (
                    f"scale=trunc(iw*{zoom}/2)*2:trunc(ih*{zoom}/2)*2,"
                    f"crop=iw/{zoom}:ih/{zoom}:"
                    f"x='(iw-iw/{zoom})*(t/{d:.3f})*0.5':"
                    f"y='(ih-ih/{zoom})*(t/{d:.3f})*0.5'"
                ),
                "-t",
                f"{d:.3f}",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(kb_tmp),
            ]
        )
        combined = mux_video_with_narration_only(kb_tmp, narr, output_path)
        try:
            kb_tmp.unlink()
        except OSError:
            pass
    elif mix_ambient and has_audio_stream(src):
        combined = combine_narration_with_ltx_ambient(src, narr, output_path)
    else:
        combined = mux_video_with_narration_only(src, narr, output_path)

    if tmp is not None:
        try:
            tmp.unlink()
        except OSError:
            pass
    return combined


def assemble_program(
    segment_videos: Sequence[str | Path],
    output_path: str | Path,
    *,
    transition_duration: float = 0.8,
) -> Path:
    """Concatenate ordered segment clips with cross-fades (video + audio)."""
    paths = [Path(p) for p in segment_videos]
    out = Path(output_path)
    td = transition_duration
    for p in paths:
        if probe_duration_seconds(p) <= td:
            raise ValueError(
                f"Segment {p.name} is shorter than transition ({td}s); "
                "shorten cross-fade or lengthen content."
            )
    return stitch_with_crossfade(
        paths,
        out,
        include_audio=True,
        transition_duration=td,
    )
