import numpy as np
import pytest
import soundfile as sf


def test_sentence_segmentation_and_no_fake_alignment():
    from timeline.models import segments_from_alignments

    words = [
        dict(text="你好。", start=0, end=3),
        dict(text="你晓得", start=5, end=6),
        dict(text="吗？", start=6, end=8),
        dict(text="再见。", start=10, end=14),
    ]
    segments = segments_from_alignments(words, 14)
    assert [(s.start, s.end, s.original_text) for s in segments] == [
        (0, 3, "你好。"),
        (5, 8, "你晓得吗？"),
        (10, 14, "再见。"),
    ]
    assert len({s.id for s in segments}) == 3
    assert segments_from_alignments([], 14) == []


def test_ranges_and_text_validation():
    from pydantic import ValidationError

    from timeline.models import SegmentEdit, validate_segments

    for start, end in [(2, 1), (float("nan"), 2), (0, float("inf"))]:
        with pytest.raises(ValidationError):
            SegmentEdit(
                start=start,
                end=end,
                original_text="原话",
                target_text="新话",
                reference_start=0,
                reference_end=1,
                reference_text="原话",
            )
    s = SegmentEdit(
        start=0,
        end=3,
        original_text="原话",
        target_text="新话",
        reference_start=0,
        reference_end=3,
        reference_text="原话",
    )
    with pytest.raises(ValueError, match="范围"):
        validate_segments([s], 2)


def test_reference_cache_identity_and_eviction(tmp_path):
    from tts.reference import PromptCache, reference_key

    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    base = dict(
        job_id="job",
        segment_id="s1",
        model="qwen:test:v1",
        source=source,
        start=0,
        end=3,
        text="你好",
        global_reference=None,
        options={"icl": True},
    )
    key = reference_key(**base)
    for change in [
        dict(segment_id="s2"),
        dict(start=1),
        dict(text="再见"),
        dict(model="v2"),
        dict(global_reference={"start": 5}),
    ]:
        assert reference_key(**(base | change)) != key
    cache = PromptCache(max_items=2)
    cache.put(key, "first", "job")
    assert cache.get(key) == "first"
    cache.put("b", "second", "other")
    cache.put("c", "third", "other")
    assert cache.get(key) is None
    cache.clear_job("other")
    assert len(cache) == 0


def test_timeline_positions_and_replacement(tmp_path):
    from timeline.audio import export_timeline

    sr = 24000
    source = tmp_path / "source.wav"
    sf.write(source, np.full(sr * 14, 0.1), sr)
    segments = []
    for i, (start, end) in enumerate([(0, 3), (5, 8), (10, 14)]):
        path = tmp_path / f"{i}.wav"
        sf.write(path, np.full(round(sr * (end - start)), 0.2 + i * 0.1), sr)
        segments.append(dict(start=start, end=end, path=path))
    for replace in [False, True]:
        out = tmp_path / f"timeline-{replace}.wav"
        export_timeline(source, 14, segments, out, replace=replace)
        a, rate = sf.read(out)
        assert rate == sr and len(a) == sr * 14
        assert np.allclose(a[sr * 3 : sr * 5], 0.1 if replace else 0, atol=1e-4)
        assert np.allclose(a[sr * 8 : sr * 10], 0.1 if replace else 0, atol=1e-4)
        assert abs(a[sr * 6] - 0.3) < 1e-4
        assert abs(a[sr * 11] - 0.4) < 1e-4


def test_duration_fit_limits_and_mp3(tmp_path):
    from timeline.audio import convert_audio, fit_audio

    path = tmp_path / "raw.wav"
    sr = 24000
    sf.write(path, np.sin(np.arange(sr * 4) * 2 * np.pi * 440 / sr) * 0.2, sr)
    result = fit_audio(path, tmp_path / "fit.wav", 3.5, "fit")
    assert result["adjusted_duration"] == pytest.approx(3.5, abs=0.02)
    assert result["tempo_ratio"] == pytest.approx(4 / 3.5, abs=0.01)
    result = fit_audio(path, tmp_path / "too-short.wav", 1, "fit")
    assert result["adjusted_duration"] == pytest.approx(4, abs=0.02)
    assert result["warning"]
    mp3 = tmp_path / "output.mp3"
    convert_audio(tmp_path / "fit.wav", mp3)
    assert mp3.stat().st_size > 1000


def test_timeline_rejects_overflow_and_overlap(tmp_path):
    from timeline.audio import export_timeline

    source = tmp_path / "source.wav"
    long = tmp_path / "long.wav"
    sf.write(source, np.zeros(24000 * 8), 24000)
    sf.write(long, np.ones(24000 * 4) * 0.2, 24000)
    with pytest.raises(ValueError, match="超出"):
        export_timeline(
            source, 8, [dict(start=0, end=3, path=long)], tmp_path / "out.wav"
        )
    with pytest.raises(ValueError, match="重叠"):
        export_timeline(
            source,
            8,
            [dict(start=0, end=4, path=long), dict(start=3, end=7, path=long)],
            tmp_path / "out.wav",
        )


def test_stereo_replacement_preserves_unmodified_channels(tmp_path):
    from timeline.audio import export_timeline

    source = tmp_path / "stereo.wav"
    segment = tmp_path / "seg.wav"
    output = tmp_path / "out.wav"
    sf.write(
        source,
        np.column_stack([np.ones(24000 * 4) * 0.1, np.ones(24000 * 4) * 0.4]),
        24000,
    )
    sf.write(segment, np.ones(24000) * 0.2, 24000)
    export_timeline(
        source, 4, [dict(start=1, end=2, path=segment)], output, replace=True
    )
    audio, sr = sf.read(output, always_2d=True)
    assert audio.shape == (24000 * 4, 2)
    assert np.allclose(audio[:24000], [0.1, 0.4], atol=1e-4)
    assert np.allclose(audio[24000:48000], [0.2, 0.2], atol=1e-4)


def test_alignment_punctuation_and_finite_clamping():
    from types import SimpleNamespace as Obj

    from asr.worker import serialize_alignments

    result = Obj(
        text="你好！再见。",
        time_stamps=Obj(
            items=[
                Obj(text=t, start_time=i * 0.5, end_time=(i + 1) * 0.5)
                for i, t in enumerate("你好再见")
            ]
        ),
    )
    items = serialize_alignments(result, 4)
    assert "".join(a["text"] for a in items) == "你好！再见。"
    assert (
        len(
            __import__(
                "timeline.models", fromlist=["segments_from_alignments"]
            ).segments_from_alignments(items, 4)
        )
        == 2
    )
    invalid = Obj(
        text="",
        time_stamps=[
            dict(text="坏", start=6, end=7),
            dict(text="坏", start=float("nan"), end=2),
        ],
    )
    assert serialize_alignments(invalid, 4) == []


def test_english_aligned_words_keep_spaces():
    from timeline.models import segments_from_alignments

    segments = segments_from_alignments(
        [dict(text="Hello", start=0, end=0.3), dict(text="world.", start=0.3, end=1)], 1
    )
    assert segments[0].original_text == "Hello world."


@pytest.mark.parametrize("excess_frames", [1, 120])
def test_even_small_overflow_cannot_touch_next_slot(tmp_path, excess_frames):
    from timeline.audio import export_timeline

    source = tmp_path / "source.wav"
    generated = tmp_path / "generated.wav"
    sf.write(source, np.ones(48000) * 0.1, 24000)
    sf.write(generated, np.ones(24000 + excess_frames) * 0.4, 24000)
    with pytest.raises(ValueError, match="超出"):
        export_timeline(
            source,
            2,
            [dict(start=0, end=1, path=generated)],
            tmp_path / "replacement.wav",
            replace=True,
        )


def test_even_submillisecond_overlaps_are_rejected(tmp_path):
    from timeline.audio import export_timeline

    source = tmp_path / "source.wav"
    generated = tmp_path / "generated.wav"
    sf.write(source, np.zeros(48000), 24000)
    sf.write(generated, np.ones(12000) * 0.1, 24000)
    with pytest.raises(ValueError, match="重叠"):
        export_timeline(
            source,
            2,
            [
                dict(start=0, end=1, path=generated),
                dict(start=0.9995, end=2, path=generated),
            ],
            tmp_path / "out.wav",
        )


def test_ascii_full_stops_split_adjacent_sentences():
    from timeline.models import segments_from_alignments

    parts = segments_from_alignments(
        [dict(text="Hello.", start=0, end=1), dict(text="Goodbye.", start=1.2, end=2)],
        2,
    )
    assert [s.original_text for s in parts] == ["Hello.", "Goodbye."]
