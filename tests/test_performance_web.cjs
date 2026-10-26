// Synthetic DOM only. Does not contact USB or microphone.
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const root = path.resolve(__dirname, "../scripts/demo_dashboard_web");
const page = fs.readFileSync(path.join(root, "performance.html"), "utf8");
const source = fs.readFileSync(path.join(root, "performance.js"), "utf8");
const nodes = new Map();
for (const match of page.matchAll(/id="([^"]+)"/g)) {
  assert(!nodes.has("#" + match[1]));
  nodes.set("#" + match[1], {textContent: "", hidden: true, addEventListener() {}});
}
for (const match of source.matchAll(/(?:\$|put)\("(#[^"]+)"/g)) assert(nodes.has(match[1]), match[1]);
let state = {connection: "NOT_CHECKED", performance: {memory_snapshot: null, last_query: null}};
let tick;
const calls = [];
vm.runInNewContext(source, {
  document: {querySelector: id => nodes.get(id)},
  window: {setInterval(cb) {tick = cb; return 1;}, clearInterval() {}, addEventListener() {}},
  fetch: async (url, options) => {
    calls.push([url, options]);
    return {ok: true, json: async () => url === "/api/session" ? {csrf_token: "synthetic"} : state};
  },
});
const flush = () => new Promise(resolve => setImmediate(resolve));
(async () => {
  await flush();
  const text = id => nodes.get("#" + id).textContent;
  assert.equal(text("heap-free"), "—");
  assert.equal(text("encoder-time"), "—");
  assert(calls.every(([url]) => url === "/api/state" || url === "/api/session"), "page load never opens USB/mic");
  const mem = {internal_heap_total_bytes: 102400, internal_heap_free_bytes: 51200,
    internal_heap_min_free_bytes: 40960, internal_heap_largest_free_block_bytes: 30720,
    recognition_us: 54000, query_heap_free_before_bytes: 51200,
    query_heap_free_after_bytes: 51200, query_heap_delta_bytes: 0};
  state = {connection: "ONLINE", performance: {memory_snapshot: mem, sampled_at: "2026-10-20T10:00:00Z",
    last_query: {job_id: "job-1", sampled_at: "2026-10-20T10:00:00Z", source: "Mic", top1: "<An>",
      metrics: {encoder_invoke_us: 53500, usb_roundtrip_us: 250000,
        arena_used_bytes: 38140, arena_capacity_bytes: 131072, input_bytes: 11920, output_bytes: 192,
        request_id: 12, telemetry: mem}}}};
  await tick();
  assert.equal(text("heap-free"), "50.0 KiB");
  assert.equal(text("encoder-time"), "53.50 ms");
  assert.equal(text("recognition-time"), "54.00 ms");
  assert.equal(text("usb-time"), "250.00 ms");
  assert(text("query-delta").startsWith("0 B"));
  assert(text("query-source").includes("<An>"), "only textContent, no HTML injection");
  state.connection = "OFFLINE";
  state.last_error = "USB timeout";
  state.active_job = {label: "new", phase: "CONNECTING"};
  await tick();
  assert(nodes.get("#refresh-board").disabled);
  assert.equal(text("connection"), "USB chưa phản hồi");
  assert(text("query-source").includes("trước đó"));
  assert(text("query-id").includes("job-1"));
  state = {performance: {last_query: {job_id: "old-fw", metrics: {encoder_invoke_us: 53500}}}};
  await tick();
  assert.equal(text("heap-free"), "—");
  assert.equal(text("recognition-time"), "—");
  assert.equal(text("encoder-time"), "53.50 ms");
  console.log("Performance render guards passed (synthetic DOM).");
})().catch(error => {console.error(error); process.exitCode = 1;});
