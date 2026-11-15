from pathlib import Path


def test_timeline_controls_and_script_loaded():
    html = Path("web/index.html").read_text()
    for element in [
        "modeSelect",
        "timelineEditor",
        "timelineCards",
        "saveTimeline",
        "generateSelected",
        "exportMode",
        "exportFormat",
        "globalSpeaker",
        "splitDialog",
    ]:
        assert f'id="{element}"' in html
    assert 'src="/web/timeline.js"' in html
    assert Path("web/timeline.js").is_file()


def test_javascript_editor_behaviors():
    import shutil
    import subprocess

    import pytest

    if not shutil.which("node"):
        pytest.skip("Node.js not installed")
    result = subprocess.run(
        ["node", "--test", "tests/timeline_ui.test.cjs"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
