"""Optional real Chromium UI smoke test; no model/network inference."""

import os
import shutil
import socket
import threading
import time

import pytest
from test_api import env as env


def test_browser_edit_generate_export_restore(env, tmp_path):
    if os.environ.get("RUN_BROWSER_TESTS") != "1":
        pytest.skip("Set RUN_BROWSER_TESTS=1 in a desktop-capable environment")
    chromium = (
        os.environ.get("CHROMIUM_EXECUTABLE_PATH")
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
    )
    if not chromium:
        pytest.skip("Chromium not installed")
    sync_api = pytest.importorskip("playwright.sync_api")
    import uvicorn

    a, client, jid, d, backend = env
    import json

    a.RECENT_FILE.write_text(json.dumps({"job_id": jid}))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(a.app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, daemon=True
    )
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.01)
    errors = []
    try:
        with sync_api.sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=chromium, headless=True, args=["--no-sandbox"]
            )
            page = browser.new_page(viewport={"width": 1100, "height": 900})
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{port}")
            page.locator("#timelineCards .segment-card").nth(2).wait_for()
            cards = page.locator("#timelineCards .segment-card")
            for i in [0, 2]:
                cards.nth(i).locator("input[type=checkbox]").first.uncheck()
            cards.nth(1).get_by_label("新台词", exact=True).fill("只改第二句")
            page.get_by_role("button", name="生成选中片段", exact=True).click()
            sync_api.expect(page.locator("#timelineNotice")).to_have_text(
                "所选片段生成完成，可以逐句试听并导出。", timeout=15000
            )
            assert len(backend.calls) == 1 and backend.calls[0][0] == "只改第二句"
            page.reload()
            sync_api.expect(
                page.locator("#timelineCards .segment-card")
                .nth(1)
                .get_by_label("新台词", exact=True)
            ).to_have_value("只改第二句")
            page.locator("#exportMode").select_option("single")
            page.get_by_role("button", name="导出所选音频", exact=True).click()
            sync_api.expect(page.locator("#timelineOutput")).to_be_visible(
                timeout=15000
            )
            download = page.locator("#timelineDownload").get_attribute("href")
            assert page.request.get(f"http://127.0.0.1:{port}" + download).ok
            page.locator("#modeSelect").select_option("free")
            sync_api.expect(page.locator("#scriptText")).to_be_visible()
            page.locator("#modeSelect").select_option("timeline")
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate(
                "document.documentElement.scrollWidth<=window.innerWidth"
            )
            assert not errors, errors
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
