const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
class Element {
  constructor(tag = "div") {
    this.tag = tag;
    this.children = [];
    this.listeners = {};
    this.style = {};
    this.dataset = {};
    this.value = "";
    this.className = "";
    this.disabled = false;
    this.classList = { toggle() {}, add() {}, remove() {} };
  }
  append(...nodes) {
    this.children.push(...nodes);
  }
  prepend(...nodes) {
    this.children.unshift(...nodes);
  }
  replaceChildren(...nodes) {
    this.children = nodes;
  }
  setAttribute() {}
  addEventListener(event, handler) {
    this.listeners[event] = handler;
  }
  querySelector(query) {
    return this.children.find((n) => n.tag === query) || null;
  }
}
function setup() {
  const elements = new Map();
  const $ = (id) => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  $("modeSelect").value = "timeline";
  const context = vm.createContext({
    $,
    jobId: "job",
    duration: 3,
    alignments: [],
    rangeStart: 0,
    rangeEnd: 1,
    window: { addEventListener() {} },
    document: {
      createElement: (t) => new Element(t),
      querySelectorAll: () => [],
      querySelector: () => null,
    },
    crypto: { randomUUID: () => String(Math.random()) },
    localStorage: {
      setItem() {},
      removeItem() {},
      getItem() {
        return null;
      },
    },
    api: async () => ({ segments: [], revision: 0, global_reference: null }),
    errText: (e) => e.message,
    confirm: () => true,
    console,
  });
  vm.runInContext(fs.readFileSync("web/timeline.js", "utf8"), context);
  return { context, $ };
}
test("incomplete numeric drafts stay editable after re-render", () => {
  const { context, $ } = setup();
  vm.runInContext(
    `timelineSegments=[{id:'s',start:null,end:3,original_text:'原',target_text:'新',reference_start:0,reference_end:3,reference_text:'原',selected:true,duration_strategy:'fit',status:'pending'}];renderTimeline();`,
    context,
  );
  assert.equal($("timelineCards").children.length, 1);
  vm.runInContext(
    `timelineSegments[0].start=0;timelineSegments[0].end=null;renderTimeline();`,
    context,
  );
  assert.equal($("timelineCards").children.length, 1);
});

test("busy state covers controls rebuilt during save", async () => {
  const { context, $ } = setup();
  await vm.runInContext(
    `action(async()=>{acceptTimeline({segments:[],revision:1,global_reference:null});if(!$('timelineFields').disabled)throw new Error('fieldset must stay disabled across render');})`,
    context,
  );
  assert.equal($("timelineError").textContent, "");
  assert.equal($("timelineFields").disabled, false);
});

test("free references snap to whole aligned words with spaces", () => {
  const html = fs.readFileSync("web/index.html", "utf8");
  const code = html.slice(
    html.indexOf("function syncReferencePhrase()"),
    html.indexOf("$('syncPhraseBtn').addEventListener"),
  );
  const ref = { value: "" };
  const context = vm.createContext({
    alignments: [
      { text: "Hello", start: 0, end: 0.5 },
      { text: "world.", start: 0.6, end: 1 },
    ],
    duration: 2,
    rangeStart: 0.2,
    rangeEnd: 0.8,
    $: (id) => (id === "modeSelect" ? { value: "free" } : ref),
    saveDraft() {},
  });
  vm.runInContext(
    "function setRange(start,end){rangeStart=start;rangeEnd=end;syncReferencePhrase();}",
    context,
  );
  vm.runInContext(code + ";syncReferencePhrase();", context);
  assert.equal(context.rangeStart, 0);
  assert.equal(context.rangeEnd, 1);
  assert.equal(ref.value, "Hello world.");
});

test("editing new text immediately marks old audio stale", () => {
  const { context } = setup();
  vm.runInContext(
    `timelineSegments=[{id:'s',status:'ready',target_text:'新话'}];timelineDraft(timelineSegments[0]);`,
    context,
  );
  assert.equal(vm.runInContext("timelineSegments[0].status", context), "stale");
});
