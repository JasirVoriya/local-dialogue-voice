from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def value(obj: Any, *names: str, default=None):
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def serialize_alignments(result: Any, duration: float) -> list[dict[str, Any]]:
    raw = value(result, "time_stamps", "timestamps", default=[]) or []
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]
    items: list[dict[str, Any]] = []
    for item in raw:
        text = str(value(item, "text", "token", default="")).strip()
        start = value(item, "start_time", "start", default=None)
        end = value(item, "end_time", "end", default=None)
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            text, start, end = str(item[0]).strip(), item[1], item[2]
        try:
            start_f, end_f = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if text and end_f > start_f:
            items.append({"text": text, "start": max(0.0, start_f), "end": min(duration, end_f)})
    return items


def load_model(asr_path: Path, aligner_path: Path, device: str, with_aligner: bool):
    import torch
    from qwen_asr import Qwen3ASRModel

    kwargs = {
        "dtype": torch.float32,
        "device_map": device,
        "max_inference_batch_size": 1,
        "max_new_tokens": 4096,
    }
    if with_aligner:
        kwargs["forced_aligner"] = str(aligner_path)
        kwargs["forced_aligner_kwargs"] = {"dtype": torch.float32, "device_map": device}
    return Qwen3ASRModel.from_pretrained(str(asr_path), **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--asr-model", required=True)
    parser.add_argument("--aligner-model", required=True)
    parser.add_argument("--duration", required=True, type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    asr_path, aligner_path = Path(args.asr_model), Path(args.aligner_model)
    try:
        import torch
        devices = ["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"]
        attempts = [(device, True) for device in devices] + [(device, False) for device in devices]
        errors = []
        output = None
        model = None
        for device, with_aligner in attempts:
            try:
                status = "ASR and timestamp model" if with_aligner else "ASR only"
                print(f"Loading {status} on {device}…", file=sys.stderr, flush=True)
                model = load_model(asr_path, aligner_path, device, with_aligner=with_aligner)
                print("Transcribing source audio…", file=sys.stderr, flush=True)
                results = model.transcribe(
                    audio=args.audio, language=None, return_time_stamps=with_aligner,
                )
                if not results:
                    raise RuntimeError("没有识别到台词。")
                result = results[0]
                text = str(value(result, "text", default="")).strip()
                if not text:
                    raise RuntimeError("没有识别到台词。")
                alignments = serialize_alignments(result, args.duration) if with_aligner else []
                output = {"transcript_text": text, "alignments": alignments,
                          "alignment_available": bool(alignments)}
                break
            except Exception as exc:
                errors.append(f"{device} {'with aligner' if with_aligner else 'without aligner'}: {type(exc).__name__}: {exc}")
                model = None
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        if output is None:
            raise RuntimeError("；".join(errors[-4:]))
        out_path = Path(args.output)
        temp_path = out_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(out_path)
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
