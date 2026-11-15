"""FFmpeg duration adaptation and disk-backed fixed-position mixing."""

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def ffmpeg(args):
    p = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *map(str, args),
        ],
        capture_output=True,
        text=True,
    )
    if p.returncode:
        raise ValueError("音频处理失败：" + p.stderr[-500:])


def convert_audio(source: Path, output: Path):
    if output.suffix == ".mp3":
        ffmpeg(
            [
                "-i",
                source,
                "-ar",
                "24000",
                "-ac",
                "1",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "192k",
                output,
            ]
        )
    else:
        ffmpeg(["-i", source, "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", output])


def fit_audio(raw: Path, output: Path, target: float, strategy: str):
    info = sf.info(raw)
    duration = info.frames / info.samplerate
    if duration <= 0:
        raise ValueError("模型生成了空音频，请重新生成。")
    ratio = duration / target
    warning = ""
    applied = 1.0
    if strategy != "natural" and abs(duration - target) > 0.01:
        if 0.8 <= ratio <= 1.25:
            applied = ratio
            ffmpeg(
                [
                    "-i",
                    raw,
                    "-af",
                    f"atempo={ratio:.9f},apad,atrim=duration={target:.9f}",
                    "-ar",
                    "24000",
                    "-ac",
                    "1",
                    output,
                ]
            )
        else:
            warning = f"时长差异过大（需要 {ratio:.2f} 倍语速），已保留自然时长；请调整文案或时间范围。"
    if applied == 1.0:
        convert_audio(raw, output)
    adjusted = sf.info(output).duration
    return dict(
        generated_duration=duration,
        adjusted_duration=adjusted,
        tempo_ratio=applied,
        warning=warning,
    )


def export_timeline(
    source: Path, duration: float, segments: list, output: Path, replace=False
):
    sr = 24000
    total = round(duration * sr)
    ordered = sorted(segments, key=lambda s: s["start"])
    last = 0
    for s in ordered:
        start, stop = round(s["start"] * sr), round(s["end"] * sr)
        if start < last:
            raise ValueError("所选片段时间范围重叠，请先调整范围。")
        if start < 0 or stop > total or stop <= start:
            raise ValueError("片段超出原视频范围或短于一个采样点。")
        info = sf.info(s["path"])
        if info.samplerate != sr or info.channels != 1:
            raise ValueError("生成片段采样率无效，请重新生成。")
        if info.frames > stop - start:
            raise ValueError("新台词超出原句范围，请缩短文案或调整时长后再导出时间轴。")
        last = stop
    # Write and overwrite on disk, never allocate the whole video in unified RAM.
    channels = sf.info(source).channels if replace else 1
    with sf.SoundFile(
        output, "w+", samplerate=sr, channels=channels, subtype="PCM_16", format="WAV"
    ) as out:
        if replace:
            with sf.SoundFile(source) as src:
                if src.samplerate != sr:
                    raise ValueError("原音轨采样率无效，请重新导入。")
                for block in src.blocks(
                    blocksize=sr * 10, dtype="float32", always_2d=True
                ):
                    remaining = total - out.tell()
                    if remaining <= 0:
                        break
                    out.write(block[:remaining])
        while out.tell() < total:
            out.write(
                np.zeros((min(sr * 10, total - out.tell()), channels), dtype=np.float32)
            )
        for s in ordered:
            start = round(s["start"] * sr)
            stop = round(s["end"] * sr)
            out.seek(start)
            left = stop - start
            while left:
                n = min(left, sr * 10)
                out.write(np.zeros((n, channels), dtype=np.float32))
                left -= n
            out.seek(start)
            with sf.SoundFile(s["path"]) as audio:
                if audio.samplerate != sr or audio.channels != 1:
                    raise ValueError("生成片段采样率无效，请重新生成。")
                for block in audio.blocks(blocksize=sr * 10, dtype="float32"):
                    out.write(np.repeat(block[:, None], channels, axis=1))


def export_sequence(segments, output):
    with sf.SoundFile(
        output, "w", samplerate=24000, channels=1, subtype="PCM_16"
    ) as out:
        for s in sorted(segments, key=lambda s: s["start"]):
            with sf.SoundFile(s["path"]) as audio:
                if audio.samplerate != 24000 or audio.channels != 1:
                    raise ValueError("生成片段格式无效。")
                for block in audio.blocks(blocksize=240000, dtype="float32"):
                    out.write(block)
