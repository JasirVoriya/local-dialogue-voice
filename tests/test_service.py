import numpy as np
import pytest
import soundfile as sf

from timeline.models import segments_from_alignments
from tts.reference import PromptCache


class FakeBackend:
    identity = "fake:v1"

    def __init__(self):
        self.references = []
        self.calls = []

    def load(self):
        return self

    def prepare_reference(self, ref):
        audio, sr = sf.read(ref.audio_path)
        prompt = (ref.text, float(np.mean(audio)), str(ref.audio_path))
        self.references.append(prompt)
        return prompt

    def synthesize(self, text, reference, path):
        self.calls.append((text, reference))
        sf.write(path, np.full(24000 * 3, 0.2), 24000)
        return 24000


def fixture_job(tmp_path):
    sr = 24000
    sf.write(
        tmp_path / "source_tts.wav",
        np.concatenate([np.full(sr * 3, x) for x in [0.1, 0.3, 0.5]]),
        sr,
    )
    segments = segments_from_alignments(
        [
            dict(text=t, start=i * 3, end=i * 3 + 3)
            for i, t in enumerate(["一。", "二。", "三。"])
        ],
        9,
    )
    return dict(
        job_id="job",
        duration=9,
        alignments=[],
        dialogue_segments=[s.model_dump() for s in segments],
        global_reference=None,
    )


def test_per_sentence_prompt_and_single_regeneration(tmp_path):
    from timeline.service import generate_segments

    job = fixture_job(tmp_path)
    backend = FakeBackend()
    cache = PromptCache()
    ids = [s["id"] for s in job["dialogue_segments"]]
    generate_segments(job, tmp_path, ids, backend, cache, lambda j: None)
    assert [r[0] for r in backend.references] == ["一。", "二。", "三。"]
    assert [r[1] for r in backend.references] == pytest.approx(
        [0.1, 0.3, 0.5], abs=0.001
    )
    assert len({s["reference_key"] for s in job["dialogue_segments"]}) == 3
    files = [tmp_path / s["generated_audio_path"] for s in job["dialogue_segments"]]
    untouched = [files[i].read_bytes() for i in [0, 2]]
    mtimes = [files[i].stat().st_mtime_ns for i in [0, 2]]
    job["dialogue_segments"][1]["target_text"] = "新的第二句"
    generate_segments(job, tmp_path, [ids[1]], backend, cache, lambda j: None)
    assert len(backend.references) == 3
    assert [files[i].read_bytes() for i in [0, 2]] == untouched
    assert [files[i].stat().st_mtime_ns for i in [0, 2]] == mtimes
    assert len(backend.calls) == 4 and backend.calls[-1][0] == "新的第二句"
    job["dialogue_segments"][1].update(
        reference_start=3.5, reference_end=5.5, reference_text="二。"
    )
    generate_segments(job, tmp_path, [ids[1]], backend, cache, lambda j: None)
    assert len(backend.references) == 4


def test_reference_change_requires_review_and_invalidates_only_changed(tmp_path):
    from timeline.service import update_segments

    job = fixture_job(tmp_path)
    from timeline.models import SegmentEdit

    edits = [
        SegmentEdit(**{k: v for k, v in s.items() if k in SegmentEdit.model_fields})
        for s in job["dialogue_segments"]
    ]
    job["dialogue_segments"][0].update(status="ready", generated_audio_path="first.wav")
    edits[1].reference_start = 3.4
    update_segments(job, edits, None)
    assert job["dialogue_segments"][1]["reference_confirmed"] is False
    assert job["dialogue_segments"][0]["generated_audio_path"] == "first.wav"
    assert job["dialogue_segments"][0]["status"] == "ready"


def test_silence_fallback_never_crosses_unknown_speakers(tmp_path):
    from timeline.service import generate_segments

    job = fixture_job(tmp_path)
    a, sr = sf.read(tmp_path / "source_tts.wav")
    a[: sr * 3] = 0
    sf.write(tmp_path / "source_tts.wav", a, sr)
    job["global_reference"] = dict(
        start=6, end=9, text="三。", speaker_id="alice", confirmed=True
    )
    first = job["dialogue_segments"][0]
    backend = FakeBackend()
    generate_segments(
        job, tmp_path, [first["id"]], backend, PromptCache(), lambda j: None
    )
    assert first["status"] == "error" and not backend.calls
    first["speaker_id"] = "alice"
    generate_segments(
        job, tmp_path, [first["id"]], backend, PromptCache(), lambda j: None
    )
    assert first["status"] == "ready"
    assert backend.references[-1][0] == "三。"
    assert first["reference_used"]["kind"] == "global_fallback"


def test_unconfirmed_reference_not_synthesized(tmp_path):
    from timeline.service import generate_segments

    job = fixture_job(tmp_path)
    s = job["dialogue_segments"][0]
    s["reference_confirmed"] = False
    backend = FakeBackend()
    generate_segments(job, tmp_path, [s["id"]], backend, PromptCache(), lambda j: None)
    assert not backend.calls and s["status"] == "error"


def test_original_correction_updates_same_range_reference(tmp_path):
    from timeline.models import SegmentEdit
    from timeline.service import update_segments

    job = fixture_job(tmp_path)
    edits = [
        SegmentEdit(**{k: v for k, v in s.items() if k in SegmentEdit.model_fields})
        for s in job["dialogue_segments"]
    ]
    edits[0].original_text = "校正后的原话"
    update_segments(job, edits, None)
    assert job["dialogue_segments"][0]["reference_text"] == "校正后的原话"
    assert not job["dialogue_segments"][0]["reference_confirmed"]
