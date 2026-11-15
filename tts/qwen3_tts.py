"""Qwen3-TTS 0.1.1 Base API. Torch/model imports remain lazy."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import TTSBackend


class Qwen3Backend(TTSBackend):
    def __init__(self, model_dir: Path, model_id: str):
        self.model_dir = model_dir
        self.model_id = model_id
        self.model = None

    @property
    def identity(self):
        try:
            package_version = version("qwen-tts")
        except Exception:
            package_version = "uninstalled"
        files = []
        if self.model_dir.exists():
            for p in sorted(self.model_dir.rglob("*")):
                if p.is_file() and (
                    p.suffix in {".json", ".safetensors"}
                    or p.name == ".download-complete"
                ):
                    st = p.stat()
                    files.append(
                        (str(p.relative_to(self.model_dir)), st.st_size, st.st_mtime_ns)
                    )
        return (
            f"{self.model_id}:qwen-tts-{package_version}:"
            + hashlib.sha256(json.dumps(files).encode()).hexdigest()
        )

    def load(self):
        if self.model is None:
            import torch
            from qwen_tts import Qwen3TTSModel

            self.model = Qwen3TTSModel.from_pretrained(
                str(self.model_dir),
                device_map="mps" if torch.backends.mps.is_available() else "cpu",
                dtype=torch.float32,
                attn_implementation="eager",
            )
        return self

    def prepare_reference(self, reference):
        return self.load().model.create_voice_clone_prompt(
            ref_audio=str(reference.audio_path),
            ref_text=reference.text,
            x_vector_only_mode=False,
        )

    def synthesize(self, text, reference, output_path):
        import torch

        with torch.inference_mode():
            wavs, sr = self.load().model.generate_voice_clone(
                text=text,
                language="Auto",
                voice_clone_prompt=reference,
                non_streaming_mode=True,
                do_sample=True,
                temperature=0.8,
                top_p=0.92,
                top_k=45,
                max_new_tokens=max(768, min(2048, len(text) * 24)),
            )
        sf.write(output_path, np.asarray(wavs[0], dtype=np.float32).reshape(-1), sr)
        del wavs
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return int(sr)

    def unload(self):
        if self.model is None:
            return
        import gc

        import torch

        self.model = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def capabilities(self):
        return dict(
            name="Qwen3-TTS Base",
            backend="qwen3",
            local=True,
            voice_clone=True,
            independent_emotion_reference=False,
            native_duration=False,
            reference_strategy="逐句音频+准确原话；全局参考仅用于同说话人回退",
            limitations="不保证方言、情绪或停顿精确复现；没有自动说话人识别。",
        )
