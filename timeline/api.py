"""HTTP integration with the existing single-background-task runner."""

import shutil
import uuid
from typing import Literal

from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import Field

from .audio import convert_audio, export_sequence, export_timeline
from .models import Reference, SegmentEdit, StrictModel
from .service import generate_segments, merge_segments, split_segment, update_segments


class SaveRequest(StrictModel):
    segments: list[SegmentEdit] = Field(max_length=1000)
    global_reference: Reference | None = None
    revision: int = Field(ge=0)


class Selection(StrictModel):
    segment_ids: list[str] = Field(min_length=1, max_length=1000)


class ExportRequest(Selection):
    mode: Literal["single", "sequence", "silence", "replace"]
    format: Literal["wav", "mp3"] = "wav"


class SplitRequest(StrictModel):
    segment_id: str
    at: float = Field(ge=0)
    left_original: str = Field(min_length=1, max_length=4000)
    right_original: str = Field(min_length=1, max_length=4000)
    left_target: str = Field(min_length=1, max_length=4000)
    right_target: str = Field(min_length=1, max_length=4000)
    revision: int = Field(ge=0)


class MergeRequest(Selection):
    revision: int = Field(ge=0)


def register_routes(app, host):
    prefix = "/api/jobs/{job_id}/timeline"

    def state(job):
        return dict(
            segments=job["dialogue_segments"],
            global_reference=job.get("global_reference"),
            revision=job.get("timeline_revision", 0),
        )

    def revision(job, expected):
        if job.get("timeline_revision", 0) != expected:
            raise HTTPException(409, "项目已在其他页面更新，请刷新后再编辑。")

    def clean_artifacts(job, d):
        # Only server-owned timeline files of removed records; model/free files untouched.
        valid = {s["id"] for s in job["dialogue_segments"]}
        for directory in ["timeline", "references"]:
            root = d / directory
            expected = {root / f"{sid}.wav" for sid in valid}
            if directory == "timeline":
                expected.update(root / "raw" / f"{sid}.wav" for sid in valid)
            if root.exists():
                for p in root.rglob("*.wav"):
                    if p not in expected:
                        p.unlink()
        shutil.rmtree(d / "exports", ignore_errors=True)

    def persist(job):
        job["timeline_revision"] = job.get("timeline_revision", 0) + 1
        clean_artifacts(job, host._job_dir(job["job_id"]))
        host._write_job(job)
        return state(job)

    def selected(job, ids):
        chosen = [s for s in job["dialogue_segments"] if s["id"] in ids]
        if len(set(ids)) != len(chosen) or not chosen:
            raise HTTPException(400, "请选择有效的台词片段。")
        return chosen

    def begin(job_id, func, body):
        host._read_job(job_id)
        task = str(uuid.uuid4())
        host._claim_task(task)
        host._start_task(task, func, job_id, body)
        return dict(task_id=task)

    @app.get("/api/tts/capabilities")
    def capabilities():
        from tts.registry import create_backend

        return create_backend(
            "qwen3", model_dir=host.TTS_MODEL_DIR, model_id=host.TTS_MODEL_ID
        ).capabilities()

    @app.post("/api/tts/unload")
    def unload():
        with host._mutation_guard():
            host._unload_tts()
        return dict(ok=True)

    @app.get(prefix)
    def get(job_id: str):
        return state(host._read_job(job_id))

    @app.put(prefix)
    def save(job_id: str, body: SaveRequest):
        with host._mutation_guard():
            job = host._read_job(job_id)
            revision(job, body.revision)
            try:
                update_segments(job, body.segments, body.global_reference)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            return persist(job)

    @app.post(prefix + "/split")
    def split(job_id: str, body: SplitRequest):
        with host._mutation_guard():
            job = host._read_job(job_id)
            revision(job, body.revision)
            try:
                split_segment(
                    job,
                    body.segment_id,
                    body.at,
                    body.left_original,
                    body.right_original,
                    body.left_target,
                    body.right_target,
                )
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            return persist(job)

    @app.post(prefix + "/merge")
    def merge(job_id: str, body: MergeRequest):
        with host._mutation_guard():
            job = host._read_job(job_id)
            revision(job, body.revision)
            try:
                merge_segments(job, body.segment_ids)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            return persist(job)

    def run_generate(task_id, job_id, body):
        job = host._read_job(job_id)
        d = host._job_dir(job_id)
        selected(job, body.segment_ids)
        shutil.rmtree(d / "exports", ignore_errors=True)
        backend = host._get_tts_backend(task_id)
        return generate_segments(
            job,
            d,
            body.segment_ids,
            backend,
            host.PROMPT_CACHE,
            host._write_job,
            lambda p, m: host._set_task(task_id, progress=p, message=m),
        )

    @app.post(prefix + "/generate")
    def generate(job_id: str, body: Selection):
        selected(host._read_job(job_id), body.segment_ids)
        return begin(job_id, run_generate, body)

    @app.get(prefix + "/{segment_id}/audio")
    def audio(job_id: str, segment_id: str):
        job = host._read_job(job_id)
        s = next((s for s in job["dialogue_segments"] if s["id"] == segment_id), None)
        if not s or s["status"] != "ready":
            raise HTTPException(404, "本句尚未生成或已被修改，请重新生成。")
        path = host._job_dir(job_id) / s["generated_audio_path"]
        if not path.is_file():
            raise HTTPException(404, "找不到生成音频，请重新生成。")
        return FileResponse(path, media_type="audio/wav")

    def run_export(task_id, job_id, body):
        job = host._read_job(job_id)
        d = host._job_dir(job_id)
        chosen = selected(job, body.segment_ids)
        if any(
            s["status"] != "ready" or not s.get("generated_audio_path") for s in chosen
        ):
            raise ValueError("所选片段尚未生成、生成失败或已修改，请先重新生成。")
        items = [
            dict(start=s["start"], end=s["end"], path=d / s["generated_audio_path"])
            for s in chosen
        ]
        folder = d / "exports"
        folder.mkdir(exist_ok=True)
        stem = uuid.uuid4().hex
        wav = folder / f"{stem}.wav"
        if body.mode == "single":
            if len(items) != 1:
                raise ValueError("单句导出只能选择一句。")
            convert_audio(items[0]["path"], wav)
        elif body.mode == "sequence":
            export_sequence(items, wav)
        else:
            source = d / "source_mix.wav"
            if not source.exists():
                video = next(d.glob("source.*"), None)
                if video:
                    from .audio import ffmpeg

                    ffmpeg(
                        [
                            "-i",
                            video,
                            "-vn",
                            "-ac",
                            "2",
                            "-ar",
                            "24000",
                            "-c:a",
                            "pcm_s16le",
                            source,
                        ]
                    )
                else:
                    # Old projects may have only extracted audio. No new source is invented.
                    source = d / "source_tts.wav"
            export_timeline(
                source, job["duration"], items, wav, replace=body.mode == "replace"
            )
        out = wav
        if body.format == "mp3":
            out = folder / f"{stem}.mp3"
            convert_audio(wav, out)
            wav.unlink()
        # Bound accumulated exports; files are task-local, not model files.
        files = sorted(folder.iterdir(), key=lambda p: p.stat().st_mtime_ns)
        for p in files[:-12]:
            p.unlink()
        return dict(
            download_url=f"/api/jobs/{job_id}/timeline/exports/{out.name}",
            filename=out.name,
        )

    @app.post(prefix + "/export")
    def export(job_id: str, body: ExportRequest):
        return begin(job_id, run_export, body)

    @app.get(prefix + "/exports/{filename}")
    def download(job_id: str, filename: str):
        import re

        if not re.fullmatch(r"[a-f0-9]{32}\.(wav|mp3)", filename):
            raise HTTPException(400, "导出文件名无效。")
        path = host._job_dir(job_id) / "exports" / filename
        if not path.is_file():
            raise HTTPException(404, "导出已过期，请重新导出。")
        return FileResponse(
            path,
            media_type="audio/wav" if path.suffix == ".wav" else "audio/mpeg",
            filename=filename,
        )
