from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VoiceReference:
    audio_path: Path
    text: str


class TTSBackend(ABC):
    @abstractmethod
    def load(self): ...
    @abstractmethod
    def prepare_reference(self, reference: VoiceReference) -> Any: ...
    @abstractmethod
    def synthesize(self, text: str, reference: Any, output_path: Path) -> int: ...
    @abstractmethod
    def unload(self): ...
    @abstractmethod
    def capabilities(self) -> dict: ...
    @property
    @abstractmethod
    def identity(self) -> str: ...
