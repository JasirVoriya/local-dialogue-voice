"""Versioned, local timeline records. No inferred speaker diarization."""

from __future__ import annotations

import re
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Reference(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str = Field(min_length=1, max_length=4000)
    speaker_id: str | None = Field(default=None, max_length=80)
    confirmed: bool = False

    @model_validator(mode="after")
    def valid_range(self):
        if self.end <= self.start or self.end - self.start > 30:
            raise ValueError("参考范围须为 0–30 秒内的连续有效片段。")
        if not self.text.strip():
            raise ValueError("参考文字不能为空。")
        return self


class SegmentEdit(StrictModel):
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), pattern=r"^[a-zA-Z0-9_-]{1,80}$"
    )
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    original_text: str = Field(max_length=4000)
    target_text: str = Field(max_length=4000)
    speaker_id: str | None = Field(default=None, max_length=80)
    reference_start: float = Field(ge=0)
    reference_end: float = Field(gt=0)
    reference_text: str = Field(max_length=4000)
    reference_confirmed: bool = True
    duration_strategy: Literal["natural", "fit", "custom"] = "fit"
    custom_duration: float | None = Field(default=None, gt=0, le=600)
    selected: bool = True

    @model_validator(mode="after")
    def valid_range(self):
        if self.end <= self.start or self.reference_end <= self.reference_start:
            raise ValueError("片段与参考范围的结束时间必须大于开始时间。")
        if self.reference_end - self.reference_start > 30:
            raise ValueError("参考范围不能超过 30 秒，请选择一句清晰原声。")
        if self.duration_strategy == "custom" and self.custom_duration is None:
            raise ValueError("自定义时长不能为空。")
        return self


class DialogueSegment(SegmentEdit):
    source_audio: str = "source_tts.wav"
    generated_audio_path: str | None = None
    raw_audio_path: str | None = None
    generated_duration: float | None = None
    adjusted_duration: float | None = None
    tempo_ratio: float | None = None
    overflow: bool = False
    status: Literal["pending", "generating", "ready", "error", "stale"] = "pending"
    warning: str = ""
    error: str = ""
    reference_key: str | None = None
    reference_used: dict | None = None


def validate_segments(segments, duration):
    if len(segments) > 1000:
        raise ValueError("单个项目最多支持 1000 个台词片段。")
    ids = set()
    for s in segments:
        if s.end > duration + 0.001 or s.reference_end > duration + 0.001:
            raise ValueError("片段或参考范围超出原视频范围。")
        if s.id in ids:
            raise ValueError("片段编号不能重复。")
        ids.add(s.id)


def join_text(tokens):
    text = ""
    for token in tokens:
        if (
            text
            and re.search(r"[A-Za-z0-9]$", text)
            and re.match(r"[A-Za-z0-9]", token)
        ):
            text += " "
        text += token
    return text


def text_for_range(alignments, start, end):
    """Only whole aligned tokens; never claim a partial token is matched."""
    selected = [
        a for a in alignments if a["start"] >= start - 0.001 and a["end"] <= end + 0.001
    ]
    partial = any(
        a["end"] > start + 0.001 and a["start"] < end - 0.001 and a not in selected
        for a in alignments
    )
    return join_text(a["text"] for a in selected), partial


def segments_from_alignments(alignments, duration):
    valid = []
    for a in alignments:
        try:
            start, end = float(a["start"]), float(a["end"])
            if 0 <= start < end <= duration and str(a["text"]).strip():
                valid.append(dict(start=start, end=end, text=str(a["text"])))
        except (KeyError, ValueError, TypeError):
            continue
    valid.sort(key=lambda a: a["start"])
    result = []
    group = []

    def emit():
        if not group:
            return
        text = join_text(a["text"] for a in group)
        start, end = group[0]["start"], max(a["end"] for a in group)
        # A long unpunctuated run must be manually split before reference use.
        ref_end = min(end, start + 30)
        ref_text, partial = text_for_range(group, start, ref_end)
        result.append(
            DialogueSegment(
                start=start,
                end=end,
                original_text=text,
                target_text=text,
                reference_start=start,
                reference_end=ref_end,
                reference_text=ref_text,
                reference_confirmed=not partial and end - start <= 30,
            )
        )

    for a in valid:
        if group and a["start"] - group[-1]["end"] >= 0.5:
            emit()
            group = []
        group.append(a)
        if re.search(r'[.。！？!?；;][”’"\']?$', a["text"]):
            emit()
            group = []
    emit()
    return result


def migrate_job(job):
    """Add timeline records lazily; never map old free scripts to guessed times."""
    if "dialogue_segments" not in job:
        job["dialogue_segments"] = [
            s.model_dump()
            for s in segments_from_alignments(
                job.get("alignments", []), job["duration"]
            )
        ]
        job["global_reference"] = None
    job["schema_version"] = 2
    return job
