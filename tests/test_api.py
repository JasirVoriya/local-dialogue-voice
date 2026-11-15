import time
import uuid
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from test_service import FakeBackend


@pytest.fixture
def env(tmp_path, monkeypatch):
    import app as a

    monkeypatch.setattr(a, "JOBS_DIR", tmp_path / "jobs")
    a.JOBS_DIR.mkdir()
    monkeypatch.setattr(a, "OUTPUT_DIR", tmp_path / "outputs")
    a.OUTPUT_DIR.mkdir()
    monkeypatch.setattr(a, "RECENT_FILE", tmp_path / "recent.json")
    monkeypatch.setattr(a, "ACTIVE_TASK_ID", None)
    a.TASKS.clear()
    fake = FakeBackend()
    monkeypatch.setattr(a, "_get_tts_backend", lambda task_id: fake)
    jid = str(uuid.uuid4())
    d = a.JOBS_DIR / jid
    d.mkdir()
    sf.write(d / "source_tts.wav", np.ones(24000 * 14) * 0.2, 24000)
    sf.write(d / "source_asr.wav", np.ones(16000 * 14) * 0.2, 16000)
    words = [
        dict(text=t, start=s, end=e)
        for t, s, e in [("一。", 0, 3), ("二。", 5, 8), ("三。", 10, 14)]
    ]
    job = dict(
        job_id=jid,
        source_name="test.mp4",
        duration=14,
        alignments=words,
        transcript_text="一。二。三。",
        transcribed=True,
        lines=[],
        gaps=[],
    )
    a._write_job(job)
    with TestClient(a.app, base_url="http://127.0.0.1") as client:
        yield a, client, jid, d, fake


def editable(segments):
    from timeline.models import SegmentEdit

    return [
        {k: v for k, v in s.items() if k in SegmentEdit.model_fields} for s in segments
    ]


def wait(client, task):
    for _ in range(200):
        state = client.get("/api/tasks/" + task).json()
        if state["status"] != "running":
            assert state["status"] == "complete", state
            return state["result"]
        time.sleep(0.01)
    pytest.fail("task timed out")


def test_api_migration_save_generate_export_and_restore(env):
    a, c, jid, d, fake = env
    url = f"/api/jobs/{jid}/timeline"
    state = c.get(url).json()
    assert len(state["segments"]) == 3
    edits = editable(state["segments"])
    edits[1]["target_text"] = "新的第二句"
    response = c.put(
        url,
        json=dict(segments=edits, global_reference=None, revision=state["revision"]),
    )
    assert response.status_code == 200, response.text
    assert (
        c.put(
            url,
            json=dict(
                segments=edits, global_reference=None, revision=state["revision"]
            ),
        ).status_code
        == 409
    )
    ids = [s["id"] for s in edits]
    r = c.post(url + "/generate", json={"segment_ids": ids})
    assert r.status_code == 200, r.text
    result = wait(c, r.json()["task_id"])
    assert not result["failed"]
    assert c.get(url).json()["segments"][1]["target_text"] == "新的第二句"
    assert c.get(url + f"/{ids[1]}/audio").status_code == 200
    for mode in ["sequence", "silence", "replace"]:
        r = c.post(url + "/export", json=dict(segment_ids=ids, mode=mode, format="mp3"))
        result = wait(c, r.json()["task_id"])
        assert c.get(result["download_url"]).content[:3] in [b"ID3", b"\xff\xfb\xc0"]
    assert c.get(f"/api/jobs/{jid}/state").json()["dialogue_segments"]
    assert c.get("/api/jobs/../../etc/passwd/timeline").status_code != 200


def test_api_reference_review_split_merge_and_busy_lock(env):
    a, c, jid, d, fake = env
    url = f"/api/jobs/{jid}/timeline"
    state = c.get(url).json()
    edits = editable(state["segments"])
    edits[1]["reference_start"] = 5.2
    state = c.put(
        url,
        json=dict(segments=edits, global_reference=None, revision=state["revision"]),
    ).json()
    assert state["segments"][1]["reference_confirmed"] is False
    s = state["segments"][0]
    r = c.post(
        url + "/split",
        json=dict(
            segment_id=s["id"],
            at=1.5,
            left_original="左",
            right_original="右",
            left_target="新左",
            right_target="新右",
            revision=state["revision"],
        ),
    )
    assert r.status_code == 200, r.text
    state = r.json()
    assert len(state["segments"]) == 4
    r = c.post(
        url + "/merge",
        json=dict(
            segment_ids=[s["id"] for s in state["segments"][:2]],
            revision=state["revision"],
        ),
    )
    assert r.status_code == 200, r.text
    assert len(r.json()["segments"]) == 3
    a.ACTIVE_TASK_ID = "busy"
    assert (
        c.put(
            url,
            json=dict(
                segments=editable(r.json()["segments"]),
                global_reference=None,
                revision=r.json()["revision"],
            ),
        ).status_code
        == 409
    )
    assert c.delete(f"/api/jobs/{jid}").status_code == 409
    a.ACTIVE_TASK_ID = None


def test_free_mode_reference_is_only_selected_clip_and_cleanup(env):
    a, c, jid, d, fake = env
    body = dict(
        job_id=jid,
        reference_start=5,
        reference_end=8,
        reference_text="二。",
        transcript_text="整段不能拼入参考",
        script_text="新一\n新二",
    )
    wait(c, c.post("/api/generate", json=body).json()["task_id"])
    assert fake.references[0][0] == "二。"
    assert sf.info(fake.references[0][2]).duration == pytest.approx(3)
    first = (d / "segment_000.wav").stat().st_mtime_ns
    wait(
        c,
        c.post(
            "/api/regenerate", json=dict(job_id=jid, line_index=1, text="再改第二句")
        ).json()["task_id"],
    )
    assert (d / "segment_000.wav").stat().st_mtime_ns == first
    assert len(fake.references) == 1
    assert c.get(f"/api/jobs/{jid}/download").status_code == 200
    assert c.delete(f"/api/jobs/{jid}").status_code == 200
    assert not d.exists()


def test_untrusted_generated_paths_and_invalid_ranges_rejected(env):
    a, c, jid, d, fake = env
    url = f"/api/jobs/{jid}/timeline"
    state = c.get(url).json()
    edits = editable(state["segments"])
    edits[0]["generated_audio_path"] = "/etc/passwd"
    assert (
        c.put(
            url,
            json=dict(
                segments=edits, global_reference=None, revision=state["revision"]
            ),
        ).status_code
        == 422
    )
    edits = editable(state["segments"])
    edits[0]["end"] = 100
    assert (
        c.put(
            url,
            json=dict(
                segments=edits, global_reference=None, revision=state["revision"]
            ),
        ).status_code
        == 400
    )
    assert c.get("/", headers={"Host": "evil.example"}).status_code == 403


def test_real_video_upload_and_asr_worker_boundary(env, tmp_path, monkeypatch):
    a, c, jid, d, fake = env
    import json
    import subprocess

    video = tmp_path / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x90:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(video),
        ],
        check=True,
    )
    response = c.post(
        "/api/upload", files={"file": ("test.mp4", video.read_bytes(), "video/mp4")}
    )
    assert response.status_code == 200, response.text
    new_id = response.json()["job_id"]
    new_dir = a._job_dir(new_id)
    for path in ["source_asr.wav", "source_tts.wav", "source_mix.wav"]:
        assert sf.info(new_dir / path).duration > 1.9
    assert c.get(f"/api/jobs/{new_id}/video").status_code == 200
    assert c.post("/api/upload", files={"file": ("bad.exe", b"bad")}).status_code == 400
    assert c.post("/api/upload", files={"file": ("empty.mp4", b"")}).status_code == 400
    # Exercise actual worker IPC/persistence boundary, without claiming neural ASR.
    root = tmp_path / "worker-app"
    (root / "asr/.venv/bin").mkdir(parents=True)
    (root / "asr/.venv/bin/python").touch()
    monkeypatch.setattr(a, "APP_DIR", root)
    monkeypatch.setattr(a, "_unload_tts", lambda: None)
    monkeypatch.setattr(a, "_safe_model_download", lambda *args: None)

    def worker_result(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps(
                dict(
                    transcript_text="合成测试。",
                    alignments=[dict(start=0, end=1.5, text="合成测试。")],
                    alignment_available=True,
                )
            )
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(a.subprocess, "run", worker_result)
    result = a._run_transcription("test", new_id)
    assert result["alignments"]
    state = c.get(f"/api/jobs/{new_id}/timeline").json()
    assert len(state["segments"]) == 1
    assert state["segments"][0]["reference_text"] == "合成测试。"


def test_interrupted_status_recovery_and_download_marker(env, tmp_path, monkeypatch):
    a, c, jid, d, fake = env
    job = a._read_job(jid)
    job["dialogue_segments"][0]["status"] = "generating"
    a._write_job(job)
    a.ACTIVE_TASK_ID = "running"
    assert a._read_job(jid)["dialogue_segments"][0]["status"] == "generating"
    a.ACTIVE_TASK_ID = None
    assert a._read_job(jid)["dialogue_segments"][0]["status"] == "error"
    model = tmp_path / "model"
    model.mkdir()
    (model / ".download-complete").touch()
    monkeypatch.setattr(
        a.subprocess,
        "run",
        lambda *a, **k: pytest.fail("must not download cached model"),
    )
    a._safe_model_download("id", model, "test", "模型", 5)


def test_validation_errors_are_chinese_and_model_unload_is_guarded(env):
    a, c, jid, d, fake = env
    response = c.post(
        "/api/generate",
        json=dict(
            job_id=jid,
            reference_start=-1,
            reference_end=0,
            reference_text="原",
            script_text="新",
        ),
    )
    assert response.status_code == 422
    assert "输入" in response.json()["detail"]
    a.ACTIVE_TASK_ID = "busy"
    assert c.post("/api/tts/unload").status_code == 409
    a.ACTIVE_TASK_ID = None
    assert c.post("/api/tts/unload").status_code == 200


def test_active_task_exposes_job_for_page_recovery(env):
    a, c, jid, d, fake = env
    assert c.get("/api/active-task").json()["task_id"] is None
    a.ACTIVE_TASK_ID = "recover-me"
    a.TASKS["recover-me"] = {"status": "running", "job_id": jid, "progress": 20}
    response = c.get("/api/active-task").json()
    assert response == {"task_id": "recover-me", "job_id": jid}
    a.ACTIVE_TASK_ID = None


def test_valid_suffix_ids_do_not_collide_or_get_deleted(env):
    a, c, jid, d, fake = env
    url = f"/api/jobs/{jid}/timeline"
    state = c.get(url).json()
    edits = editable(state["segments"])[:2]
    edits[0]["id"] = "foo"
    edits[1]["id"] = "foo-raw"
    state = c.put(
        url,
        json=dict(segments=edits, global_reference=None, revision=state["revision"]),
    ).json()
    wait(
        c,
        c.post(url + "/generate", json=dict(segment_ids=["foo", "foo-raw"])).json()[
            "task_id"
        ],
    )
    state = c.get(url).json()
    assert (
        state["segments"][0]["raw_audio_path"]
        != state["segments"][1]["generated_audio_path"]
    )
    c.put(
        url,
        json=dict(
            segments=editable(state["segments"]),
            global_reference=None,
            revision=state["revision"],
        ),
    )
    for segment_id in ["foo", "foo-raw"]:
        assert c.get(url + f"/{segment_id}/audio").status_code == 200
