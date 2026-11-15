"""Timeline editing/generation, independent of HTTP and real model loading."""

import uuid
from pathlib import Path

from tts.reference import crop_reference, reference_key

from .audio import fit_audio
from .models import (
    DialogueSegment,
    text_for_range,
    validate_segments,
)


def update_segments(job, edits, global_reference):
    validate_segments(edits, job["duration"])
    if global_reference and global_reference.end > job["duration"] + 0.001:
        raise ValueError("全局参考范围超出视频范围。")
    previous = {s["id"]: s for s in job["dialogue_segments"]}
    global_data = global_reference.model_dump() if global_reference else None
    global_changed = job.get("global_reference") != global_data
    updated = []
    for edit in edits:
        data = edit.model_dump()
        old = previous.get(edit.id)
        changed = not old or any(data[k] != old.get(k) for k in data if k != "selected")
        segment = DialogueSegment(**data).model_dump()
        if old:
            segment = old | data
        ref_changed = old and (
            old["reference_start"] != edit.reference_start
            or old["reference_end"] != edit.reference_end
        )
        if ref_changed:
            text, partial = text_for_range(
                job.get("alignments", []), edit.reference_start, edit.reference_end
            )
            segment["reference_text"] = text
            segment["reference_confirmed"] = False
            segment["warning"] = "参考范围已变更，请试听并校对原话后勾选确认。" + (
                "范围切到了字词中间，建议对齐字词边界。" if partial else ""
            )
        if (
            old
            and old["original_text"] != edit.original_text
            and edit.reference_start == edit.start
            and edit.reference_end == edit.end
        ):
            segment["reference_text"] = edit.original_text
            segment["reference_confirmed"] = False
        if changed or global_changed:
            segment.update(
                status="stale"
                if old and old.get("generated_audio_path")
                else "pending",
                error="",
            )
        updated.append(segment)
    job["dialogue_segments"] = sorted(updated, key=lambda s: s["start"])
    job["global_reference"] = global_data


def prepare_prompt(job, segment, job_dir, backend, cache):
    if not segment["reference_confirmed"]:
        raise ValueError("请先校对并确认本句参考音频与原话严格对应。")
    source = job_dir / "source_tts.wav"
    start, end, text = (
        segment["reference_start"],
        segment["reference_end"],
        segment["reference_text"],
    )
    ref_dir = job_dir / "references"
    path = ref_dir / f"{segment['id']}.wav"
    kind = "local"
    warning = ""
    try:
        ref = crop_reference(source, start, end, text, path)
    except ValueError as exc:
        global_ref = job.get("global_reference")
        if not (
            global_ref
            and global_ref.get("confirmed")
            and segment.get("speaker_id")
            and segment["speaker_id"] == global_ref.get("speaker_id")
        ):
            raise ValueError(str(exc) + " 未设置已确认的同一说话人全局参考。") from exc
        start, end, text = global_ref["start"], global_ref["end"], global_ref["text"]
        ref = crop_reference(source, start, end, text, path)
        kind = "global_fallback"
        warning = (
            "局部参考质量不足，已使用同说话人的全局音色参考，局部语气可能无法保留。"
        )
    key = reference_key(
        job_id=job["job_id"],
        segment_id=segment["id"],
        model=backend.identity,
        source=source,
        start=start,
        end=end,
        text=text,
        global_reference=job.get("global_reference"),
        options={
            "x_vector_only_mode": False,
            "strategy": kind,
            "local_range": [segment["reference_start"], segment["reference_end"]],
            "local_text": segment["reference_text"],
            "speaker_id": segment.get("speaker_id"),
        },
    )
    prompt = cache.get(key)
    if prompt is None:
        prompt = backend.prepare_reference(ref)
        cache.put(key, prompt, job["job_id"])
    return prompt, key, dict(kind=kind, start=start, end=end, text=text), warning


def generate_segments(
    job, job_dir: Path, ids, backend, cache, save, progress=lambda *args: None
):
    selected = [s for s in job["dialogue_segments"] if s["id"] in ids]
    if not selected or len(selected) != len(set(ids)):
        raise ValueError("请选择有效的台词片段。")
    if any(not s["target_text"].strip() for s in selected):
        raise ValueError("新台词不能为空；如需保留原话，请使用恢复原话。")
    out_dir = job_dir / "timeline"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "raw").mkdir(exist_ok=True)
    for i, s in enumerate(selected):
        progress(
            20 + int(i / len(selected) * 70),
            f"正在生成时间轴第 {i + 1}/{len(selected)} 句…",
        )
        s.update(status="generating", error="")
        save(job)
        token = uuid.uuid4().hex
        raw = out_dir / f".{token}-raw.wav"
        adjusted = out_dir / f".{token}-fit.wav"
        try:
            prompt, key, used, warning = prepare_prompt(job, s, job_dir, backend, cache)
            backend.synthesize(s["target_text"].strip(), prompt, raw)
            target = (
                s["custom_duration"]
                if s["duration_strategy"] == "custom"
                else s["end"] - s["start"]
            )
            metrics = fit_audio(raw, adjusted, target, s["duration_strategy"])
            raw_final = out_dir / "raw" / f"{s['id']}.wav"
            final = out_dir / f"{s['id']}.wav"
            raw.replace(raw_final)
            adjusted.replace(final)
            s.update(metrics)
            s.update(
                status="ready",
                generated_audio_path=str(final.relative_to(job_dir)),
                raw_audio_path=str(raw_final.relative_to(job_dir)),
                reference_key=key,
                reference_used=used,
                warning=" ".join(filter(None, [warning, metrics["warning"]])),
                overflow=round(metrics["adjusted_duration"] * 24000)
                > round(s["end"] * 24000) - round(s["start"] * 24000),
            )
        except Exception as exc:
            s.update(status="error", error=str(exc)[:800])
        finally:
            raw.unlink(missing_ok=True)
            adjusted.unlink(missing_ok=True)
            save(job)
    return dict(
        job_id=job["job_id"],
        segments=job["dialogue_segments"],
        failed=[s["id"] for s in selected if s["status"] == "error"],
    )


def split_segment(
    job, segment_id, at, left_original, right_original, left_target, right_target
):
    old = next((s for s in job["dialogue_segments"] if s["id"] == segment_id), None)
    if not old or not old["start"] < at < old["end"]:
        raise ValueError("拆分位置必须在本句时间范围内部。")
    result = []
    for start, end, original, target in [
        (old["start"], at, left_original, left_target),
        (at, old["end"], right_original, right_target),
    ]:
        result.append(
            DialogueSegment(
                start=start,
                end=end,
                original_text=original,
                target_text=target,
                speaker_id=old.get("speaker_id"),
                reference_start=start,
                reference_end=min(end, start + 30),
                reference_text=original,
                reference_confirmed=False,
            ).model_dump()
        )
    index = job["dialogue_segments"].index(old)
    job["dialogue_segments"][index : index + 1] = result


def merge_segments(job, ids):
    ordered = sorted(job["dialogue_segments"], key=lambda s: s["start"])
    selected = [s for s in ordered if s["id"] in ids]
    indices = [ordered.index(s) for s in selected]
    if (
        len(selected) < 2
        or len(selected) != len(set(ids))
        or indices != list(range(indices[0], indices[-1] + 1))
    ):
        raise ValueError("请选择至少两个相邻片段合并。")
    if len({s.get("speaker_id") for s in selected}) != 1:
        raise ValueError("不能合并不同说话人的片段。")
    start, end = selected[0]["start"], selected[-1]["end"]
    if end - start > 30:
        raise ValueError("合并后的参考超过 30 秒，请保留较短的自然句。")
    original = "".join(s["original_text"] for s in selected)
    merged = DialogueSegment(
        start=start,
        end=end,
        original_text=original,
        target_text="".join(s["target_text"] for s in selected),
        speaker_id=selected[0].get("speaker_id"),
        reference_start=start,
        reference_end=end,
        reference_text=original,
        reference_confirmed=False,
    ).model_dump()
    job["dialogue_segments"] = (
        ordered[: indices[0]] + [merged] + ordered[indices[-1] + 1 :]
    )
