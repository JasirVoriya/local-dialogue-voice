"""Register only validated implementations; remote backends are disabled."""

from .qwen3_tts import Qwen3Backend

BACKENDS = {"qwen3": Qwen3Backend}


def create_backend(name, **kwargs):
    if name not in BACKENDS:
        raise ValueError("该 TTS 后端尚未验证，不能启用。")
    return BACKENDS[name](**kwargs)
