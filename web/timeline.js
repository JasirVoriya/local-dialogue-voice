/* Timeline UI extends the existing vanilla page and reuses its video/task APIs. */
let timelineSegments = [],
  timelineRevision = 0,
  timelineBusy = false,
  timelineDirty = false,
  splitId = null;
const editFields = [
  "id",
  "start",
  "end",
  "original_text",
  "target_text",
  "speaker_id",
  "reference_start",
  "reference_end",
  "reference_text",
  "reference_confirmed",
  "duration_strategy",
  "custom_duration",
  "selected",
];
function modeChanged() {
  const timelineMode = $("modeSelect").value === "timeline";
  $("stepScript").classList.toggle("hidden", timelineMode);
  $("timelineEditor").classList.toggle("hidden", !timelineMode || !jobId);
  if (
    !timelineMode &&
    jobId &&
    alignments.some(
      (a) =>
        a.end > rangeStart &&
        a.start < rangeEnd &&
        (a.start < rangeStart - 0.001 || a.end > rangeEnd + 0.001),
    )
  )
    syncReferencePhrase();
}
$("modeSelect").addEventListener("change", modeChanged);
function timelineDraft(segment) {
  if (segment && segment.status === "ready") {
    segment.status = "stale";
    const card = document.querySelector(`[data-segment-id="${segment.id}"]`);
    if (card) {
      card.querySelector("audio")?.remove();
      const metrics = card.querySelector(".segment-metrics");
      if (metrics) metrics.textContent = "本句已修改，请重新生成后试听。";
    }
  }
  $("timelineOutput").classList.add("hidden");
  timelineDirty = true;
  $("timelineNotice").textContent = "有未保存的编辑。生成和导出前会自动保存。";
  if (jobId)
    localStorage.setItem(
      `timelineDraft_${jobId}`,
      JSON.stringify({
        revision: timelineRevision,
        segments: timelineSegments.map((s) =>
          Object.fromEntries(editFields.map((k) => [k, s[k]])),
        ),
        global_reference: readGlobal(),
      }),
    );
}
function readGlobal() {
  if (!$("globalEnd").value) return null;
  return {
    start: Number($("globalStart").value),
    end: Number($("globalEnd").value),
    text: $("globalText").value,
    speaker_id: $("globalSpeaker").value.trim() || null,
    confirmed: $("globalConfirmed").checked,
  };
}
function renderGlobal(g) {
  $("globalStart").value = g?.start ?? "";
  $("globalEnd").value = g?.end ?? "";
  $("globalText").value = g?.text ?? "";
  $("globalSpeaker").value = g?.speaker_id ?? "";
  $("globalConfirmed").checked = g?.confirmed ?? false;
}
function acceptTimeline(data) {
  timelineSegments = data.segments;
  timelineRevision = data.revision;
  renderGlobal(data.global_reference);
  timelineDirty = false;
  renderTimeline();
}
async function loadTimeline(resume = true) {
  modeChanged();
  if (!jobId) {
    timelineSegments = [];
    renderTimeline();
    return;
  }
  try {
    const data = await api(`/api/jobs/${jobId}/timeline`);
    acceptTimeline(data);
    const raw = localStorage.getItem(`timelineDraft_${jobId}`);
    if (raw) {
      const draft = JSON.parse(raw);
      if (draft.revision === timelineRevision) {
        const live = new Map(data.segments.map((s) => [s.id, s]));
        timelineSegments = draft.segments.map((s) => {
          const current = live.get(s.id);
          const edited = Object.fromEntries(editFields.map((k) => [k, s[k]]));
          const changed =
            current &&
            editFields.some((k) => k !== "selected" && current[k] !== s[k]);
          return {
            ...current,
            ...edited,
            status:
              changed && current?.status === "ready"
                ? "stale"
                : (current?.status ?? "pending"),
          };
        });
        renderGlobal(draft.global_reference);
        timelineDirty = true;
        renderTimeline();
        $("timelineNotice").textContent = "已恢复未保存的编辑，请检查后保存。";
      }
    }
    const caps = await api("/api/tts/capabilities");
    $("backendInfo").textContent =
      `当前后端：${caps.name} · 本机声音克隆 · 独立情绪参考：${caps.independent_emotion_reference ? "支持" : "不支持"} · 原生时长控制：${caps.native_duration ? "支持" : "不支持"}。${caps.limitations}`;
    if (resume && !timelineBusy) {
      const active = await api("/api/active-task");
      if (active.task_id && active.job_id === jobId) {
        await action(async () => {
          await timelineTask(active.task_id);
          await restoreJob(jobId);
          await loadTimeline(false);
        });
      }
    }
  } catch (e) {
    $("timelineError").textContent = errText(e);
  }
}
window.addEventListener("jobChanged", loadTimeline);
window.addEventListener("transcriptionDone", loadTimeline);
async function saveTimeline() {
  if (!jobId) throw new Error("请先导入视频。");
  const segments = timelineSegments.map((s) =>
    Object.fromEntries(editFields.map((k) => [k, s[k]])),
  );
  const data = await api(`/api/jobs/${jobId}/timeline`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      segments,
      global_reference: readGlobal(),
      revision: timelineRevision,
    }),
  });
  acceptTimeline(data);
  localStorage.removeItem(`timelineDraft_${jobId}`);
  $("timelineNotice").textContent =
    "编辑已保存。参考范围变更后，请校对原话并重新勾选确认。";
  $("timelineOutput").classList.add("hidden");
}
async function action(fn) {
  if (timelineBusy) return;
  timelineBusy = true;
  $("timelineError").textContent = "";
  $("timelineFields").disabled = true;
  try {
    await fn();
  } catch (e) {
    $("timelineError").textContent = errText(e);
  } finally {
    timelineBusy = false;
    $("timelineFields").disabled = false;
  }
}
function selectedIds() {
  return timelineSegments.filter((s) => s.selected).map((s) => s.id);
}
function textInRange(start, end) {
  return alignments
    .filter((a) => a.start >= start - 0.001 && a.end <= end + 0.001)
    .map((a) => a.text)
    .reduce(
      (t, x) =>
        t + (/[A-Za-z0-9]$/.test(t) && /^[A-Za-z0-9]/.test(x) ? " " : "") + x,
      "",
    );
}
function markReference(s) {
  s.reference_confirmed = false;
  const card = document.querySelector(`[data-segment-id="${s.id}"]`);
  if (card) {
    const check = card.querySelector(".ref-confirmed");
    if (check) check.checked = false;
    const text = card.querySelector(".reference-text");
    if (text && document.activeElement !== text) text.value = s.reference_text;
  }
  timelineDraft(s);
}
function node(tag, text, cls) {
  const n = document.createElement(tag);
  if (text != null) n.textContent = text;
  if (cls) n.className = cls;
  return n;
}
function button(text, handler) {
  const n = node("button", text, "button");
  n.type = "button";
  n.addEventListener("click", handler);
  return n;
}
function field(label, value, onchange, type = "text") {
  const wrap = node("div", null, "field");
  const id = `field-${crypto.randomUUID()}`;
  const lab = node("label", label);
  lab.htmlFor = id;
  const input = document.createElement(
    type === "textarea" ? "textarea" : "input",
  );
  input.id = id;
  if (type === "textarea") {
    input.rows = 2;
    input.maxLength = 4000;
  } else {
    input.type = type;
    if (type === "number") {
      input.step = ".01";
      input.min = 0;
    } else input.maxLength = 80;
  }
  input.value = value ?? "";
  input.addEventListener("input", () =>
    onchange(
      type === "number"
        ? input.value === ""
          ? null
          : Number(input.value)
        : input.value,
    ),
  );
  wrap.append(lab, input);
  return wrap;
}
function playOriginal(start, end) {
  const video = $("video");
  video.currentTime = start;
  video.play().catch((e) => ($("timelineError").textContent = errText(e)));
  if (window.originalStop)
    video.removeEventListener("timeupdate", window.originalStop);
  window.originalStop = () => {
    if (video.currentTime >= end) {
      video.pause();
      video.removeEventListener("timeupdate", window.originalStop);
    }
  };
  video.addEventListener("timeupdate", window.originalStop);
}
function renderTimeline() {
  const root = $("timelineCards");
  root.replaceChildren();
  if (!timelineSegments.length) {
    root.append(
      node("p", "暂无时间轴片段。请识别原声，或按视频选区手动新增。", "muted"),
    );
    return;
  }
  timelineSegments.forEach((s, i) => {
    const card = node("article", null, "line-card segment-card");
    card.dataset.segmentId = s.id;
    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = s.selected;
    check.addEventListener("change", () => {
      s.selected = check.checked;
      timelineDraft();
    });
    const heading = node(
      "label",
      ` ${i + 1} · ${Number.isFinite(s.start) ? s.start.toFixed(2) : "待填"}–${Number.isFinite(s.end) ? s.end.toFixed(2) : "待填"} 秒`,
    );
    heading.prepend(check);
    card.append(heading);
    const times = node("div", null, "segment-grid");
    for (const [label, key] of [
      ["开始（秒）", "start"],
      ["结束（秒）", "end"],
    ])
      times.append(
        field(
          label,
          s[key],
          (v) => {
            s[key] = v;
            timelineDraft(s);
          },
          "number",
        ),
      );
    card.append(
      times,
      field(
        "原台词",
        s.original_text,
        (v) => {
          s.original_text = v;
          if (s.reference_start === s.start && s.reference_end === s.end) {
            s.reference_text = v;
            markReference(s);
          } else timelineDraft(s);
        },
        "textarea",
      ),
      field(
        "新台词",
        s.target_text,
        (v) => {
          s.target_text = v;
          timelineDraft(s);
        },
        "textarea",
      ),
      field("说话人标记（全局回退须与此一致）", s.speaker_id, (v) => {
        s.speaker_id = v.trim() || null;
        timelineDraft(s);
      }),
    );
    const refs = node("div", null, "segment-grid");
    for (const [label, key] of [
      ["参考开始（秒）", "reference_start"],
      ["参考结束（秒）", "reference_end"],
    ])
      refs.append(
        field(
          label,
          s[key],
          (v) => {
            s[key] = v;
            s.reference_text = textInRange(s.reference_start, s.reference_end);
            markReference(s);
          },
          "number",
        ),
      );
    const refTextField = field(
      "参考片段的准确原话",
      s.reference_text,
      (v) => {
        s.reference_text = v;
        markReference(s);
      },
      "textarea",
    );
    refTextField.querySelector("textarea").className = "reference-text";
    card.append(refs, refTextField);
    const confirm = node("label", " 已试听并确认参考原话与所选音频严格对应");
    const confirmed = document.createElement("input");
    confirmed.type = "checkbox";
    confirmed.className = "ref-confirmed";
    confirmed.checked = s.reference_confirmed;
    confirmed.addEventListener("change", () => {
      s.reference_confirmed = confirmed.checked;
      timelineDraft(s);
    });
    confirm.prepend(confirmed);
    card.append(confirm);
    const strategy = document.createElement("select");
    strategy.setAttribute("aria-label", "时长策略");
    for (const [v, t] of [
      ["natural", "自然生成"],
      ["fit", "适配原句"],
      ["custom", "自定义时长"],
    ]) {
      const opt = node("option", t);
      opt.value = v;
      strategy.append(opt);
    }
    strategy.value = s.duration_strategy;
    strategy.addEventListener("change", () => {
      s.duration_strategy = strategy.value;
      timelineDraft(s);
    });
    const config = node("div", null, "controls");
    config.append(
      strategy,
      field(
        "自定义时长（秒）",
        s.custom_duration,
        (v) => {
          s.custom_duration = v;
          timelineDraft(s);
        },
        "number",
      ),
    );
    card.append(config);
    const sec = (v) => (v == null ? "—" : Number(v).toFixed(2) + "s");
    const statuses = {
      pending: "待生成",
      ready: "已生成",
      stale: "已修改，需重新生成",
      error: "生成失败",
      generating: "正在生成",
    };
    card.append(
      node(
        "div",
        `原始 ${sec(s.end - s.start)} · 生成 ${sec(s.generated_duration)} · 调整后 ${sec(s.adjusted_duration)} · 实际语速 ${s.tempo_ratio?.toFixed(2) ?? "—"}× · ${s.overflow ? "超出原句范围" : "未超出原句范围"} · ${statuses[s.status] ?? s.status}`,
        "segment-metrics",
      ),
    );
    if (s.warning) card.append(node("p", s.warning, "warning"));
    if (s.error) card.append(node("p", s.error, "error"));
    const controls = node("div", null, "controls");
    controls.append(
      button("跳转到本句", () => {
        $("video").currentTime = s.start;
        $("video").scrollIntoView({ behavior: "smooth", block: "center" });
      }),
      button("试听原声", () => playOriginal(s.start, s.end)),
      button("试听参考", () =>
        playOriginal(s.reference_start, s.reference_end),
      ),
      button("采用视频选区作参考", () => {
        s.reference_start = rangeStart;
        s.reference_end = rangeEnd;
        s.reference_text = textInRange(rangeStart, rangeEnd);
        markReference(s);
        renderTimeline();
      }),
      button("恢复原始台词", () => {
        s.target_text = s.original_text;
        timelineDraft(s);
        renderTimeline();
      }),
      button("重新生成本句", () => action(() => generateTimeline([s.id]))),
      button("拆分", () => {
        splitId = s.id;
        $("splitAt").value =
          $("video").currentTime > s.start && $("video").currentTime < s.end
            ? $("video").currentTime
            : (s.start + s.end) / 2;
        for (const id of [
          "splitLeftOriginal",
          "splitRightOriginal",
          "splitLeftTarget",
          "splitRightTarget",
        ])
          $(id).value = "";
        $("splitDialog").showModal();
      }),
    );
    card.append(controls);
    if (s.status === "ready" && s.generated_audio_path) {
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "none";
      audio.src = `/api/jobs/${jobId}/timeline/${s.id}/audio?v=${Date.now()}`;
      audio.setAttribute("aria-label", `第${i + 1}句新声`);
      card.append(audio);
    }
    root.append(card);
  });
}
async function timelineTask(id) {
  $("timelineProgress").style.display = "block";
  while (true) {
    const t = await api(`/api/tasks/${id}`);
    $("timelineProgressText").textContent = t.message;
    $("timelineProgressPercent").textContent = `${t.progress}%`;
    $("timelineProgressBar").style.width = t.progress + "%";
    if (t.status === "error") throw new Error(t.error);
    if (t.status === "complete") return t.result;
    await sleep(600);
  }
}
async function generateTimeline(ids) {
  if (!ids.length) throw new Error("请至少选择一句。");
  if (timelineDirty) await saveTimeline();
  const r = await api(`/api/jobs/${jobId}/timeline/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ segment_ids: ids }),
  });
  const result = await timelineTask(r.task_id);
  await loadTimeline();
  if (result.failed?.length)
    throw new Error(
      `${result.failed.length} 句生成失败，请查看每句下方的中文错误提示。`,
    );
  $("timelineNotice").textContent = "所选片段生成完成，可以逐句试听并导出。";
}
$("saveTimeline").onclick = () => action(saveTimeline);
$("generateSelected").onclick = () =>
  action(() => generateTimeline(selectedIds()));
$("selectAllSegments").onclick = () => {
  timelineSegments.forEach((s) => (s.selected = true));
  timelineDraft();
  renderTimeline();
};
$("selectRangeSegments").onclick = () => {
  timelineSegments.forEach(
    (s) => (s.selected = s.end > rangeStart && s.start < rangeEnd),
  );
  timelineDraft();
  renderTimeline();
};
$("addSegment").onclick = () => {
  const text = textInRange(rangeStart, rangeEnd);
  timelineSegments.push({
    id: crypto.randomUUID(),
    start: rangeStart,
    end: rangeEnd,
    original_text: text,
    target_text: text,
    speaker_id: null,
    reference_start: rangeStart,
    reference_end: Math.min(rangeEnd, rangeStart + 30),
    reference_text: text,
    reference_confirmed: false,
    duration_strategy: "fit",
    custom_duration: null,
    selected: true,
    status: "pending",
  });
  timelineSegments.sort((a, b) => a.start - b.start);
  timelineDraft();
  renderTimeline();
};
$("removeSegments").onclick = () => {
  if (confirm("移除选中的时间轴片段？原视频不会删除。")) {
    timelineSegments = timelineSegments.filter((s) => !s.selected);
    timelineDraft();
    renderTimeline();
  }
};
$("useGlobalRange").onclick = () => {
  renderGlobal({
    start: rangeStart,
    end: rangeEnd,
    text: $("referenceText").value,
    speaker_id: $("globalSpeaker").value,
    confirmed: false,
  });
  timelineDraft();
};
$("clearGlobal").onclick = () => {
  renderGlobal(null);
  timelineDraft();
};
for (const id of [
  "globalStart",
  "globalEnd",
  "globalText",
  "globalSpeaker",
  "globalConfirmed",
])
  $(id).addEventListener("input", () => {
    if (id !== "globalConfirmed") $("globalConfirmed").checked = false;
    timelineSegments.forEach((s) => timelineDraft(s));
    timelineDraft();
  });
$("clearTimelineJob").onclick = () => $("clearBtn").click();
$("cancelSplit").onclick = () => $("splitDialog").close();
$("splitForm").onsubmit = (e) => {
  e.preventDefault();
  action(async () => {
    if (timelineDirty) await saveTimeline();
    const r = await api(`/api/jobs/${jobId}/timeline/split`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        segment_id: splitId,
        at: Number($("splitAt").value),
        left_original: $("splitLeftOriginal").value,
        right_original: $("splitRightOriginal").value,
        left_target: $("splitLeftTarget").value,
        right_target: $("splitRightTarget").value,
        revision: timelineRevision,
      }),
    });
    acceptTimeline(r);
    $("splitDialog").close();
  });
};
$("mergeSegments").onclick = () =>
  action(async () => {
    const ids = selectedIds();
    if (
      !confirm("确认选中的相邻台词属于同一说话人？合并后需要重新确认参考原话。")
    )
      return;
    if (timelineDirty) await saveTimeline();
    const r = await api(`/api/jobs/${jobId}/timeline/merge`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ segment_ids: ids, revision: timelineRevision }),
    });
    acceptTimeline(r);
  });
$("exportTimeline").onclick = () =>
  action(async () => {
    if (timelineDirty) await saveTimeline();
    const r = await api(`/api/jobs/${jobId}/timeline/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        segment_ids: selectedIds(),
        mode: $("exportMode").value,
        format: $("exportFormat").value,
      }),
    });
    const result = await timelineTask(r.task_id);
    $("timelinePlayer").src = result.download_url;
    $("timelineDownload").href = result.download_url;
    $("timelineDownload").download = result.filename;
    $("timelineOutput").classList.remove("hidden");
  });
window.addEventListener("beforeunload", (e) => {
  if (timelineDirty) {
    e.preventDefault();
    e.returnValue = "";
  }
});
modeChanged();

$("unloadModel").onclick = () =>
  action(async () => {
    await api("/api/tts/unload", { method: "POST" });
    $("timelineNotice").textContent =
      "模型与参考缓存已释放；下次生成会从本机重新加载。";
  });
