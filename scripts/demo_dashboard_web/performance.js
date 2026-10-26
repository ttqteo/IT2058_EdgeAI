(() => {
  "use strict";
  const $ = id => document.querySelector(id);
  const put = (id, value) => { $(id).textContent = value; };
  const number = value => typeof value === "number" && Number.isFinite(value);
  const bytes = value => number(value) ? `${(value / 1024).toFixed(1)} KiB` : "—";
  const ms = value => number(value) ? `${(value / 1000).toFixed(2)} ms` : "—";
  const date = value => value ? new Date(value).toLocaleString("vi-VN") : "chưa có";
  let csrf = null;
  let submitting = false;
  let reading = false;
  let timer = null;
  const api = async (path, options = {}) => {
    const response = await fetch(path, {cache: "no-store", ...options});
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
    return value;
  };
  const render = state => {
    const perf = state.performance || {};
    const mem = perf.memory_snapshot;
    const labels = {ONLINE: "Đã đọc ESP32-S3", OFFLINE: "USB chưa phản hồi", CONNECTING: "Đang xử lý USB", ERROR: "Tác vụ lỗi", NOT_CHECKED: "Chưa đọc board"};
    put("#connection", labels[state.connection] || "Chưa có kết nối");
    put("#snapshot-time", `Số đo RAM gần nhất · ${date(perf.sampled_at)}`);
    put("#heap-free", bytes(mem && mem.internal_heap_free_bytes));
    put("#heap-total", mem ? `Heap nội bộ quản lý · ${bytes(mem.internal_heap_total_bytes)}` : "Chưa đo · không phải tổng SRAM chip");
    put("#heap-largest", bytes(mem && mem.internal_heap_largest_free_block_bytes));
    put("#heap-min", bytes(mem && mem.internal_heap_min_free_bytes));
    put("#telemetry-note", mem ? "Phạm vi: heap INTERNAL + 8BIT. Số liệu được lấy ở thời điểm trên, không phải RAM trống liên tục." : "Chưa có telemetry từ firmware. Không suy ra RAM trống từ arena; cần bản firmware hỗ trợ.");
    const active = state.active_job;
    $("#refresh-board").disabled = submitting || Boolean(active) || !csrf;
    put("#activity", active ? `Đang chạy: ${active.label} · ${active.phase}. Chờ tác vụ này hoàn tất.` : "Chạy query ở trang Demo; trang này tự cập nhật dữ liệu cache, không tự gọi USB.");
    $("#error").hidden = !state.last_error;
    put("#error", state.last_error || "");
    const query = perf.last_query;
    const metrics = query && query.metrics;
    const telemetry = metrics && metrics.telemetry;
    put("#query-time", date(query && query.sampled_at));
    put("#query-source", query ? `${query.source} · board chọn ${query.top1 || "—"}. Đây là query thành công trước đó, không phải kết quả của tác vụ đang chạy.` : "Chạy mic hoặc file ở trang Demo để có số đo query.");
    put("#encoder-time", ms(metrics && metrics.encoder_invoke_us));
    put("#frontend-time", ms(metrics && metrics.frontend_us));
    put("#recognition-scope", metrics && metrics.operation === "RECOGNIZE_PCM" ? "Frontend + encoder + cosine; không gồm thu âm/USB" : "Tensor đối chứng: copy + encoder + cosine; không gồm frontend/USB");
    put("#recognition-time", ms(telemetry && telemetry.recognition_us));
    put("#usb-time", ms(metrics && metrics.usb_roundtrip_us));
    put("#usb-scope", metrics && metrics.operation === "RECOGNIZE_PCM" ? "Gửi PCM + frontend + encoder + cosine + nhận; không gồm thu âm" : "Gửi tensor + encoder + cosine + nhận; không gồm frontend/thu âm");
    put("#query-arena", metrics ? `${bytes(metrics.arena_used_bytes)} / ${bytes(metrics.arena_capacity_bytes)}` : "—");
    put("#query-tensors", metrics ? `${bytes(metrics.input_bytes)} / ${bytes(metrics.output_bytes)}` : "—");
    put("#query-heap", telemetry ? `${bytes(telemetry.query_heap_free_before_bytes)} → ${bytes(telemetry.query_heap_free_after_bytes)}` : "Chưa có số đo trước/sau");
    put("#query-delta", telemetry && number(telemetry.query_heap_delta_bytes) ? `${telemetry.query_heap_delta_bytes} B · không phải RAM đỉnh` : "—");
    put("#query-id", query ? `${query.job_id} · request ${metrics && metrics.request_id != null ? metrics.request_id : "chưa ghi nhận"}` : "");
  };
  const refresh = async () => {
    if (reading) return;
    reading = true;
    try { render(await api("/api/state")); }
    catch (error) {
      $("#error").hidden = false;
      put("#error", `Không đọc được dashboard: ${error.message}. Các số hiển thị cũ chưa được xác nhận lại.`);
      put("#connection", "Mất kết nối dashboard");
    } finally { reading = false; }
  };
  $("#refresh-board").addEventListener("click", async () => {
    if (!csrf || submitting) return;
    submitting = true;
    $("#refresh-board").disabled = true;
    try {
      await api("/api/action", {method: "POST", headers: {"Content-Type": "application/json", "X-CSRF-Token": csrf}, body: JSON.stringify({action: "check_board"})});
    } catch (error) {
      $("#error").hidden = false;
      put("#error", error.message);
    } finally { submitting = false; await refresh(); }
  });
  const boot = async () => {
    try { csrf = (await api("/api/session")).csrf_token; }
    catch (error) { put("#connection", `Dashboard chưa sẵn sàng: ${error.message}`); }
    await refresh();
    timer = window.setInterval(refresh, 1000);
  };
  window.addEventListener("beforeunload", () => { if (timer) window.clearInterval(timer); });
  boot();
})();
