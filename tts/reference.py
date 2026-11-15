"""Bounded memory-only prompts and exact audio/transcript reference pairs."""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import VoiceReference


def source_identity(source: Path):
    stat = source.stat()
    return [str(source.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def reference_key(
    *, job_id, segment_id, model, source, start, end, text, global_reference, options
):
    data = dict(
        job=job_id,
        segment=segment_id,
        model=model,
        source=source_identity(source),
        start=start,
        end=end,
        text=text,
        global_reference=global_reference,
        options=options,
    )
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _bytes(value):
    if isinstance(value, dict):
        return sum(_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_bytes(v) for v in value)
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return value.numel() * value.element_size()
    if hasattr(value, "nbytes"):
        return value.nbytes
    if hasattr(value, "__dict__"):
        return _bytes(vars(value))
    return 0


class PromptCache:
    def __init__(self, max_items=8, max_bytes=128 * 1024 * 1024):
        self.max_items = max_items
        self.max_bytes = max_bytes
        self._items = OrderedDict()

    def get(self, key):
        if key not in self._items:
            return None
        self._items.move_to_end(key)
        return self._items[key][0]

    def put(self, key, value, job_id):
        self._items[key] = (value, job_id, _bytes(value))
        self._items.move_to_end(key)
        while (
            len(self._items) > self.max_items
            or sum(v[2] for v in self._items.values()) > self.max_bytes
        ):
            self._items.popitem(last=False)

    def clear(self):
        self._items.clear()

    def clear_job(self, job_id):
        for key in list(self._items):
            if self._items[key][1] == job_id:
                del self._items[key]

    def __len__(self):
        return len(self._items)


def crop_reference(source: Path, start: float, end: float, text: str, output: Path):
    if not text.strip():
        raise ValueError("参考原话为空，请校对参考文字。")
    with sf.SoundFile(source) as f:
        sr = f.samplerate
        if start < 0 or end <= start or end > len(f) / sr + 0.001 or end - start > 30:
            raise ValueError("参考音频范围无效或超过 30 秒。")
        f.seek(round(start * sr))
        audio = f.read(round((end - start) * sr), dtype="float32", always_2d=True).mean(
            axis=1
        )
    if len(audio) / sr < 0.5 or not len(audio) or np.sqrt(np.mean(audio**2)) < 0.003:
        raise ValueError(
            "参考原声过短或接近静音，请重新选取清晰语音，或设置同一说话人的全局参考。"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio, sr)
    return VoiceReference(output, text.strip())
