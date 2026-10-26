// Synthetic rendering guards only; no USB, network, or microphone access.
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const root = path.resolve(__dirname, "..", "scripts", "demo_dashboard_web");
const page = fs.readFileSync(path.join(root, "index.html"), "utf8");
const source = fs.readFileSync(path.join(root, "app.js"), "utf8");
assert(page.includes("Thu PCM · 3 giây"), "capture action uses raw PCM label");
assert(page.includes("không phải mô phỏng âm học INMP441"), "acoustic scope is explicit");
assert(source.includes("recognition_us"), "board telemetry timing is rendered");
assert(source.includes("usb_roundtrip_us"), "USB timing is rendered separately");
const nodes = new Map();
function element() {
  const step = {textContent: ""};
  return {textContent: "", innerHTML: "", hidden: false, style: {}, dataset: {},
    attributes: {}, querySelector: () => step,
    setAttribute(key, value) { this.attributes[key] = value; },
    removeAttribute(key) { delete this.attributes[key]; }};
}
for (const match of page.matchAll(/id="([^"]+)"/g)) {
  assert(!nodes.has("#" + match[1]), "unique DOM IDs");
  nodes.set("#" + match[1], element());
}
// Every literal ID read by the app must exist, even inside collapsed details.
for (const match of source.matchAll(/(?:\$|text)\("#([^"]+)"/g)) {
  assert(nodes.has("#" + match[1]), "missing DOM ID: " + match[1]);
}
let state = {connection: "NOT_CHECKED", board: null, history: []};
let tick;
vm.runInNewContext(source, {
  document: {querySelector: id => nodes.get(id), querySelectorAll: () => []},
  window: {setInterval(callback) { tick = callback; return 1; }, clearInterval() {}, addEventListener() {}},
  fetch: async url => ({ok: true, json: async () => url === "/api/session" ? {csrf_token: "test"} : state}),
});
const flush = () => new Promise(resolve => setImmediate(resolve));
const job = (phase, status = "RUNNING", action = "run_live") => ({phase, status, action, progress: 40, details: {}});
(async () => {
  await flush();
  const node = id => nodes.get("#" + id);
  assert.equal(node("recording-callout").hidden, true);
  for (const phase of ["MIC_PERMISSION", "MIC_PREPARE", "MIC_RECORDING", "FRONTEND", "BOARD_INFER", "QUERY"]) {
    state = {active_job: job(phase), history: []};
    await tick();
    assert.equal(node("recording-callout").hidden, phase !== "MIC_RECORDING");
    assert.equal(node(phase === "BOARD_INFER" || phase === "QUERY" ? "flow-board" : "flow-audio").dataset.state, "active");
    assert.equal(node("results-list").hidden, true);
  }
  state = {active_job: job("STATUS_READONLY", "RUNNING", "check_board")};
  await tick();
  assert.equal(node("flow-audio").dataset.state, "idle");
  assert.equal(node("flow-result").dataset.state, "idle");
  const result = {kind: "live_mac_query", top1: "<An>", scores: ["Quang", "Tien", "An"].map(label => ({label, cosine: 0.5})), encoder_invoke_us: 53500,
    capture: {duration_seconds: 3, profile_id: "MAC_RAW_PCM16_MONO16K_V1", acoustic_equivalence: "UNVERIFIED", input_device: {name: "Synthetic input"}},
    performance: {recognition_us: null, frontend_us: 1200, encoder_us: 53500, usb_roundtrip_us: 60000}};
  state = {last_job: {...job("COMPLETE", "COMPLETED"), result}, connection: "ONLINE", last_check_at: "2026-10-20T08:00:00Z",
    board: {templates: 3, labels: ["Quang", "Tien", "An"], persistence_state: "READY"}};
  await tick();
  assert.equal(node("flow-result").dataset.state, "done");
  assert.equal(node("results-list").hidden, false);
  assert.equal((node("results-list").innerHTML.match(/class="score-row"/g) || []).length, 3);
  assert(node("results-list").innerHTML.includes("&lt;An&gt;"));
  assert(node("results-list").innerHTML.includes("53.5 ms"));
  assert(node("results-list").innerHTML.includes("Xử lý trên board · —"), "missing telemetry stays missing");
  assert(node("results-list").innerHTML.includes("Độ dài bản thu · 3.000 s"));
  assert(!node("results-list").innerHTML.includes("Đúng nhãn"));
  assert(node("templates-value").textContent.includes("Quang / Tien / An"));
  assert(node("connection-detail").textContent.includes("Lần đọc gần nhất"));
  state = {last_job: {...job("ERROR", "ERROR"), result, error: "write failed"}};
  await tick();
  assert.equal(node("results-list").hidden, true, "never show a successful result on an error job");
  assert.equal(node("recording-callout").hidden, true);
  assert.equal(node("error-box").hidden, false);
  console.log("Dashboard render guards passed (synthetic DOM).");
})().catch(error => { console.error(error); process.exitCode = 1; });
