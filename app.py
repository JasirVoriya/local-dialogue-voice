from __future__ import annotations

import asyncio
import gc
import json
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
WEB_DIR = APP_DIR / "web"
OUTPUT_DIR = APP_DIR / "outputs"
APP_DATA = Path.home() / "Library" / "Application Support" / "TaiCiPeiyin"
JOBS_DIR = APP_DATA / "jobs"
MODEL_DIR = APP_DATA / "models"
RECENT_FILE = APP_DATA / "recent_job.json"
ASR_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
ASR_MODEL_DIR = MODEL_DIR / "Qwen3-ASR-0.6B"
ALIGNER_MODEL_DIR = MODEL_DIR / "Qwen3-ForcedAligner-0.6B"
TTS_MODEL_DIR = MODEL_DIR / "Qwen3-TTS-12Hz-1.7B-Base"
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024

app = FastAPI(title="本机台词配音工具", docs_url=None, redoc_url=None)
TASKS: dict[str, dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()
MODEL_LOCK = threading.RLock()
ACTIVE_TASK_ID: str | None = None
_TTS_MODEL = None
_PROMPTS: dict[str, Any] = {}


@app.middleware("http")
async def restrict_to_localhost(request, call_next):
    host = request.headers.get("host", "").split(":", 1)[0].strip("[]").lower()
    origin = request.headers.get("origin")
    if host not in {"127.0.0.1", "localhost"}:
        return JSONResponse({"detail": "只允许从本机打开此工具。"}, status_code=403)
    if origin:
        origin_host = urlparse(origin).hostname
        if origin_host not in {"127.0.0.1", "localhost"}:
            return JSONResponse({"detail": "只允许从本机页面访问此工具。"}, status_code=403)
    return await call_next(request)


class GenerateRequest(BaseModel):
    job_id: str
    transcript_text: str = ""
    reference_start: float = Field(ge=0)
    reference_end: float = Field(gt=0)
    reference_text: str = ""
    script_text: str


class RegenerateRequest(BaseModel):
    job_id: str
    line_index: int = Field(ge=0)
    text: str


class DeleteRequest(BaseModel):
    job_id: str


def _job_dir(job_id: str) -> Path:
    try:
        normalized = str(uuid.UUID(job_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="项目编号无效。") from exc
    path = (JOBS_DIR / normalized).resolve()
    if path.parent != JOBS_DIR.resolve() or not path.is_dir():
        raise HTTPException(status_code=404, detail="找不到这个项目，请重新导入视频。")
    return path


def _read_job(job_id: str) -> dict[str, Any]:
    path = _job_dir(job_id) / "project.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="项目资料不完整，请重新导入视频。")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_job(job: dict[str, Any]) -> None:
    d = _job_dir(job["job_id"])
    temp = d / "project.json.tmp"
    temp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(d / "project.json")


def _set_task(task_id: str, *, progress: int | None = None, message: str | None = None,
              status: str | None = None, result: dict[str, Any] | None = None,
              error: str | None = None) -> None:
    with TASKS_LOCK:
        task = TASKS.setdefault(task_id, {"status": "running", "progress": 0, "message": "正在准备…"})
        if progress is not None:
            task["progress"] = max(0, min(100, progress))
        if message is not None:
            task["message"] = message
        if status is not None:
            task["status"] = status
        if result is not None:
            task["result"] = result
        if error is not None:
            task["error"] = error


def _claim_task(task_id: str) -> None:
    global ACTIVE_TASK_ID
    with TASKS_LOCK:
        if ACTIVE_TASK_ID is not None:
            raise HTTPException(status_code=409, detail="当前有任务正在运行，请等它完成后再试。")
        ACTIVE_TASK_ID = task_id
        TASKS[task_id] = {"status": "running", "progress": 1, "message": "任务已排队…"}


def _start_task(task_id: str, func, *args) -> None:
    def runner() -> None:
        global ACTIVE_TASK_ID
        try:
            result = func(task_id, *args)
            _set_task(task_id, status="complete", progress=100, message="完成。", result=result)
        except Exception as exc:
            traceback.print_exc()
            _set_task(task_id, status="error", message="处理失败。", error=str(exc)[:800])
        finally:
            with TASKS_LOCK:
                ACTIVE_TASK_ID = None
    threading.Thread(target=runner, name=f"task-{task_id[:8]}", daemon=True).start()


def _safe_model_download(model_id: str, dest: Path, task_id: str, label: str, progress: int) -> None:
    marker = dest / ".download-complete"
    if marker.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    exe = Path(sys.executable).parent / "modelscope"
    if not exe.exists():
        found = shutil.which("modelscope")
        if found:
            exe = Path(found)
    if not exe.exists():
        raise RuntimeError("找不到 ModelScope 下载程序。请重新运行启动器安装依赖。")
    _set_task(task_id, progress=progress, message=f"首次使用：正在下载{label}，请保持页面和电脑开启…")
    log_path = APP_DATA / "last-model-download.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [str(exe), "download", "--model", model_id, "--local_dir", str(dest)],
            stdout=log, stderr=subprocess.STDOUT, text=True,
        )
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-1200:]
        raise RuntimeError(f"{label}下载失败。请检查网络后重试。\n{tail}")
    marker.write_text(datetime.now().isoformat(), encoding="utf-8")


def _unload_tts() -> None:
    global _TTS_MODEL
    if _TTS_MODEL is not None:
        _TTS_MODEL = None
        _PROMPTS.clear()
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def _get_tts_model(task_id: str):
    global _TTS_MODEL
    with MODEL_LOCK:
        if _TTS_MODEL is not None:
            return _TTS_MODEL
        _safe_model_download(TTS_MODEL_ID, TTS_MODEL_DIR, task_id, "声音生成模型", 5)
        _set_task(task_id, progress=18, message="正在加载本机声音生成模型…")
        from qwen_tts import Qwen3TTSModel

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _TTS_MODEL = Qwen3TTSModel.from_pretrained(
            str(TTS_MODEL_DIR), device_map=device, dtype=torch.float32,
            attn_implementation="eager",
        )
        return _TTS_MODEL


def _extract_audio(video_path: Path, job_dir: Path, task_id: str) -> tuple[Path, Path, float]:
    if task_id != "upload":
        _set_task(task_id, progress=4, message="正在读取视频并提取原声…")
    duration_result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video_path)],
        capture_output=True, text=True,
    )
    if duration_result.returncode != 0:
        raise RuntimeError("无法读取视频时长，请确认视频文件正常。")
    duration = float(duration_result.stdout.strip())
    asr_path = job_dir / "source_asr.wav"
    tts_path = job_dir / "source_tts.wav"
    base = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path), "-vn"]
    first = subprocess.run(base + ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(asr_path)], capture_output=True, text=True)
    if first.returncode != 0 or not asr_path.is_file() or asr_path.stat().st_size < 1024:
        raise RuntimeError("视频中没有可读取的音轨。")
    second = subprocess.run(base + ["-ac", "1", "-ar", "24000", "-af", "highpass=f=80,lowpass=f=8000", "-c:a", "pcm_s16le", str(tts_path)], capture_output=True, text=True)
    if second.returncode != 0:
        raise RuntimeError("无法处理视频原声。")
    return asr_path, tts_path, duration


def _run_transcription(task_id: str, job_id: str) -> dict[str, Any]:
    job = _read_job(job_id)
    job_dir = _job_dir(job_id)
    _unload_tts()
    _safe_model_download(ASR_MODEL_ID, ASR_MODEL_DIR, task_id, "转写模型", 5)
    _safe_model_download(ALIGNER_MODEL_ID, ALIGNER_MODEL_DIR, task_id, "时间对齐模型", 12)
    _set_task(task_id, progress=24, message="正在加载本机转写模型并识别原声…")
    worker = APP_DIR / "asr" / "worker.py"
    python = APP_DIR / "asr" / ".venv" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError("转写环境尚未安装。请退出工具后重新双击启动.command。")
    output_path = job_dir / "asr_result.json"
    proc = subprocess.run(
        [str(python), str(worker), "--audio", str(job_dir / "source_asr.wav"),
         "--asr-model", str(ASR_MODEL_DIR), "--aligner-model", str(ALIGNER_MODEL_DIR),
         "--duration", str(job["duration"]), "--output", str(output_path)],
        capture_output=True, text=True, timeout=60 * 60,
    )
    if proc.returncode != 0 or not output_path.is_file():
        details = (proc.stderr or proc.stdout or "").strip()[-1600:]
        raise RuntimeError("本机转写失败。请重新启动工具后重试。\n" + details)
    data = json.loads(output_path.read_text(encoding="utf-8"))
    transcript = str(data.get("transcript_text", "")).strip()
    alignments = data.get("alignments", [])
    if not transcript:
        raise RuntimeError("没有识别到台词。可以换一段更清楚的视频再试。")
    job["transcript_text"] = transcript
    job["alignments"] = alignments
    job["alignment_available"] = bool(data.get("alignment_available", alignments))
    job["transcribed"] = True
    _write_job(job)
    return {"job_id": job_id, "transcript_text": transcript, "alignments": alignments, "duration": job["duration"]}


def _parse_script(script: str) -> tuple[list[str], list[float]]:
    raw_lines = script.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines: list[str] = []
    gaps: list[float] = []
    pending_blank = False
    for raw in raw_lines:
        text = raw.strip()
        if not text:
            if lines:
                pending_blank = True
            continue
        if lines:
            gaps.append(1.05 if pending_blank else 0.38)
        lines.append(text)
        pending_blank = False
    return lines, gaps


def _make_reference(job: dict[str, Any], job_dir: Path, start: float, end: float,
                    transcript_text: str, phrase_text: str) -> tuple[Path, str]:
    duration = float(job["duration"])
    if end <= start or start < 0 or end > duration + 0.05:
        raise RuntimeError("参考句的起止时间不正确，请重新拖选时间范围。")
    if end - start < 0.25:
        raise RuntimeError("参考句太短，请至少选取四分之一秒的原声。")
    phrase_path = job_dir / "selected_reference.wav"
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-i", str(job_dir / "source_tts.wav"),
        "-t", f"{end - start:.3f}", "-ac", "1", "-ar", "24000",
        "-c:a", "pcm_s16le", str(phrase_path),
    ]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("无法截取所选参考句，请重新选择时间范围。")
    full_audio, sr = sf.read(job_dir / "source_tts.wav", dtype="float32")
    phrase_audio, phrase_sr = sf.read(phrase_path, dtype="float32")
    if sr != 24000 or phrase_sr != sr:
        raise RuntimeError("音频采样率不匹配，请重新导入视频。")
    joined = np.concatenate((full_audio, np.zeros(round(0.35 * sr), dtype=np.float32), phrase_audio))
    combined_path = job_dir / "voice_reference.wav"
    sf.write(combined_path, joined, sr)
    base_text = transcript_text.strip()
    if not base_text:
        raise RuntimeError("原视频转写文字为空，请先转写或手动填写原声文字。")
    phrase_text = phrase_text.strip()
    if not phrase_text:
        raise RuntimeError("请填写选中参考句的原声文字。")
    combined_text = f"{base_text} {phrase_text}"
    return combined_path, combined_text


def _synthesize_line(model, prompt, text: str, output_path: Path) -> int:
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="Auto",
        voice_clone_prompt=prompt,
        non_streaming_mode=True,
        do_sample=True,
        temperature=0.8,
        top_p=0.92,
        top_k=45,
        max_new_tokens=max(768, min(2048, len(text) * 24)),
    )
    audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    sf.write(output_path, audio, sr)
    del wavs, audio
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return int(sr)


def _assemble_mp3(job: dict[str, Any], job_dir: Path) -> Path:
    paths = [job_dir / f"segment_{i:03d}.wav" for i in range(len(job.get("lines", [])))]
    if not paths or any(not p.is_file() for p in paths):
        raise RuntimeError("还有台词未生成，无法导出完整音频。")
    sample_rate = None
    parts: list[np.ndarray] = []
    gaps = job.get("gaps", [])
    for i, path in enumerate(paths):
        audio, sr = sf.read(path, dtype="float32")
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise RuntimeError("各句音频采样率不一致，无法合并。")
        parts.append(np.asarray(audio).reshape(-1))
        if i < len(paths) - 1:
            gap = gaps[i] if i < len(gaps) else 0.38
            parts.append(np.zeros(round(sample_rate * float(gap)), dtype=np.float32))
    joined = np.concatenate(parts)
    wav_path = job_dir / "complete.wav"
    sf.write(wav_path, joined, sample_rate)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = re.sub(r"[^\w\-一-龥]+", "_", Path(job["source_name"]).stem).strip("_") or "台词配音"
    if not job.get("output_path"):
        name = f"{base}_台词配音_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        job["output_path"] = str(OUTPUT_DIR / name)
    out_path = Path(job["output_path"])
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav_path),
         "-af", "loudnorm=I=-16:LRA=11:TP=-1.5", "-ar", "48000", "-ac", "1", "-codec:a", "libmp3lame", "-b:a", "192k", str(out_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("合并 MP3 失败：" + proc.stderr[-600:])
    job["duration_out"] = len(joined) / sample_rate
    _write_job(job)
    return out_path


def _run_generate(task_id: str, request: GenerateRequest) -> dict[str, Any]:
    job = _read_job(request.job_id)
    job_dir = _job_dir(request.job_id)
    lines, gaps = _parse_script(request.script_text)
    if not lines:
        raise RuntimeError("请先填写要生成的台词。")
    if not request.reference_text.strip():
        raise RuntimeError("请填写选中参考句的原声文字。")
    ref_audio, ref_text = _make_reference(
        job, job_dir, request.reference_start, request.reference_end,
        request.transcript_text, request.reference_text,
    )
    model = _get_tts_model(task_id)
    _set_task(task_id, progress=25, message="正在准备音色、口音和语气参考…")
    prompt = model.create_voice_clone_prompt(
        ref_audio=str(ref_audio), ref_text=ref_text, x_vector_only_mode=False,
    )
    _PROMPTS[request.job_id] = prompt
    job.update({
        "transcript_text": request.transcript_text,
        "reference_start": request.reference_start,
        "reference_end": request.reference_end,
        "reference_text": request.reference_text,
        "lines": lines,
        "gaps": gaps,
        "output_path": job.get("output_path"),
    })
    _write_job(job)
    sr = 24000
    for i, line in enumerate(lines):
        _set_task(task_id, progress=30 + int(i / max(1, len(lines)) * 60),
                  message=f"正在生成第 {i + 1}/{len(lines)} 句…")
        sr = _synthesize_line(model, prompt, line, job_dir / f"segment_{i:03d}.wav")
    out = _assemble_mp3(job, job_dir)
    _set_task(task_id, progress=96, message="正在整理试听和下载文件…")
    return {"job_id": request.job_id, "lines": lines, "gaps": gaps,
            "audio_url": f"/api/jobs/{request.job_id}/audio", "filename": out.name,
            "duration": job.get("duration_out", 0), "sample_rate": sr}


def _run_regenerate(task_id: str, request: RegenerateRequest) -> dict[str, Any]:
    job = _read_job(request.job_id)
    job_dir = _job_dir(request.job_id)
    lines = job.get("lines", [])
    if request.line_index >= len(lines):
        raise RuntimeError("找不到要重做的句子，请重新生成整段音频。")
    text = request.text.strip()
    if not text:
        raise RuntimeError("这句台词为空，无法生成。")
    prompt = _PROMPTS.get(request.job_id)
    model = _get_tts_model(task_id)
    if prompt is None:
        _set_task(task_id, progress=20, message="正在重新准备音色参考…")
        ref_audio, ref_text = _make_reference(
            job, job_dir, float(job["reference_start"]), float(job["reference_end"]),
            job.get("transcript_text", ""), job.get("reference_text", ""),
        )
        prompt = model.create_voice_clone_prompt(ref_audio=str(ref_audio), ref_text=ref_text, x_vector_only_mode=False)
        _PROMPTS[request.job_id] = prompt
    _set_task(task_id, progress=45, message=f"正在单独重做第 {request.line_index + 1} 句…")
    _synthesize_line(model, prompt, text, job_dir / f"segment_{request.line_index:03d}.wav")
    lines[request.line_index] = text
    job["lines"] = lines
    out = _assemble_mp3(job, job_dir)
    return {"job_id": request.job_id, "line_index": request.line_index, "text": text,
            "audio_url": f"/api/jobs/{request.job_id}/audio", "filename": out.name,
            "segment_url": f"/api/jobs/{request.job_id}/segments/{request.line_index}"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "local_only": True, "ffmpeg": bool(shutil.which("ffmpeg"))}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="请选择 MP4、MOV、M4V、MKV 或 WebM 视频。")
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / f"source{suffix}"
    total = 0
    try:
        with video_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="视频超过 2 GB，请先剪成较短片段。")
                output.write(chunk)
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    finally:
        await file.close()
    if total == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="视频文件为空。")
    try:
        asr, tts, duration = await asyncio.to_thread(_extract_audio, video_path, job_dir, "upload")
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job = {"job_id": job_id, "source_name": Path(file.filename or "视频").name,
           "source_path": str(video_path), "duration": duration,
           "source_asr": str(asr), "source_tts": str(tts), "transcribed": False,
           "transcript_text": "", "alignments": [], "lines": [], "gaps": []}
    _write_job(job)
    RECENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RECENT_FILE.write_text(json.dumps({"job_id": job_id}), encoding="utf-8")
    return {"job_id": job_id, "source_name": job["source_name"], "duration": duration,
            "video_url": f"/api/jobs/{job_id}/video"}


@app.post("/api/transcribe")
def transcribe(body: DeleteRequest) -> dict[str, str]:
    job = _read_job(body.job_id)
    if job.get("transcribed"):
        return {"job_id": body.job_id, "status": "already_done"}
    task_id = str(uuid.uuid4())
    _claim_task(task_id)
    _start_task(task_id, _run_transcription, body.job_id)
    return {"task_id": task_id}


@app.post("/api/generate")
def generate(body: GenerateRequest) -> dict[str, str]:
    _read_job(body.job_id)
    task_id = str(uuid.uuid4())
    _claim_task(task_id)
    _start_task(task_id, _run_generate, body)
    return {"task_id": task_id}


@app.post("/api/regenerate")
def regenerate(body: RegenerateRequest) -> dict[str, str]:
    _read_job(body.job_id)
    task_id = str(uuid.uuid4())
    _claim_task(task_id)
    _start_task(task_id, _run_regenerate, body)
    return {"task_id": task_id}


@app.get("/api/tasks/{task_id}")
def task_status(task_id: str) -> dict[str, Any]:
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="找不到任务。")
        return dict(task)


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str):
    job_dir = _job_dir(job_id)
    video_path = next((p for p in job_dir.glob("source.*") if p.is_file()), None)
    if not video_path:
        raise HTTPException(status_code=404, detail="找不到原视频。")
    media = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
    return FileResponse(video_path, media_type=media)


@app.get("/api/jobs/{job_id}/segments/{line_index}")
def get_segment(job_id: str, line_index: int):
    job_dir = _job_dir(job_id)
    path = job_dir / f"segment_{line_index:03d}.wav"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="这句还没有生成。")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/jobs/{job_id}/audio")
def get_audio(job_id: str):
    job = _read_job(job_id)
    path = Path(job.get("output_path", ""))
    if not path.is_file():
        raise HTTPException(status_code=404, detail="完整音频还没有生成。")
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/api/jobs/{job_id}/download")
def download_audio(job_id: str):
    job = _read_job(job_id)
    path = Path(job.get("output_path", ""))
    if not path.is_file():
        raise HTTPException(status_code=404, detail="完整音频还没有生成。")
    return FileResponse(path, media_type="audio/mpeg", filename=path.name)


@app.get("/api/jobs/{job_id}/state")
def get_job_state(job_id: str):
    job = _read_job(job_id)
    output = Path(job.get("output_path", "")) if job.get("output_path") else None
    return {
        "job_id": job_id,
        "source_name": job["source_name"],
        "duration": job["duration"],
        "video_url": f"/api/jobs/{job_id}/video",
        "transcribed": job.get("transcribed", False),
        "transcript_text": job.get("transcript_text", ""),
        "alignments": job.get("alignments", []),
        "reference_start": job.get("reference_start", 0),
        "reference_end": job.get("reference_end", min(4, job["duration"])),
        "reference_text": job.get("reference_text", ""),
        "lines": job.get("lines", []),
        "audio_ready": bool(output and output.is_file()),
        "audio_url": f"/api/jobs/{job_id}/audio",
        "filename": output.name if output else "",
    }


@app.get("/api/jobs/recent")
def get_recent_job():
    if not RECENT_FILE.is_file():
        return {"job_id": None}
    try:
        job_id = json.loads(RECENT_FILE.read_text(encoding="utf-8")).get("job_id")
        if not job_id:
            return {"job_id": None}
        _read_job(job_id)
        return {"job_id": job_id}
    except Exception:
        return {"job_id": None}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    with TASKS_LOCK:
        if ACTIVE_TASK_ID:
            raise HTTPException(status_code=409, detail="任务正在运行，完成后再清理项目。")
    job_dir = _job_dir(job_id)
    job = _read_job(job_id)
    output_path = Path(job.get("output_path", "")) if job.get("output_path") else None
    shutil.rmtree(job_dir, ignore_errors=True)
    _PROMPTS.pop(job_id, None)
    if RECENT_FILE.is_file():
        try:
            if json.loads(RECENT_FILE.read_text(encoding="utf-8")).get("job_id") == job_id:
                RECENT_FILE.unlink()
        except Exception:
            pass
    if output_path and output_path.is_file() and output_path.parent == OUTPUT_DIR.resolve():
        output_path.unlink()
    return {"ok": True}


def _open_browser(url: str) -> None:
    time.sleep(1.0)
    webbrowser.open(url)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    APP_DATA.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"本机台词配音工具已启动：{url}")
    print("处理过程只在本机运行。首次使用需要下载模型文件。")
    threading.Thread(target=_open_browser, args=(url,), daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
