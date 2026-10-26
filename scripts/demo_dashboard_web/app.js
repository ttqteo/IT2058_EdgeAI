(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const buttons = [...document.querySelectorAll("[data-action]")];
  const OFFLINE_HINT = "Không thấy USB native ESP32-S3 303A:1001, serial 7C:0C:5F:5B:0E:00. Kiểm tra cáp và cổng native USB rồi bấm Kiểm tra board.";
  let csrfToken = null;
  let polling = null;
  let resultsSuppressed = false;

  const html = (value) => String(value == null ? "" : value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");

  const text = (selector, value) => {
    const node = $(selector);
    if (node) node.textContent = value == null || value === "" ? "—" : String(value);
  };

  const escapeDate = (value) => {
    if (!value) return "—";
    try { return new Date(value).toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit", second: "2-digit" }); }
    catch (_) { return value; }
  };

  const api = async (path, options = {}) => {
    const response = await fetch(path, { cache: "no-store", ...options });
    let payload = {};
    try { payload = await response.json(); } catch (_) { payload = { error: "Phản hồi server không đọc được" }; }
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    return payload;
  };

  const connectionLabel = (state) => ({
    NOT_CHECKED: ["Chưa kiểm tra", "status-neutral"],
    CONNECTING: ["Đang kết nối", "status-connecting"],
    ONLINE: ["Đã đọc board", "status-online"],
    OFFLINE: ["Mất kết nối", "status-error"],
    ERROR: ["Tác vụ lỗi", "status-error"],
  }[state] || [state || "—", "status-neutral"]);

  const renderConnection = (state, lastCheckAt) => {
    const badge = $("#connection-badge");
    if (!badge) return;
    const [label, className] = connectionLabel(state);
    badge.textContent = label;
    badge.className = `status-badge ${className}`;
    const detail = $("#connection-detail");
    if (!detail) return;
    if (state === "ONLINE" && lastCheckAt) {
      detail.textContent = `Lần đọc gần nhất · ${escapeDate(lastCheckAt)}`;
    } else if (state === "OFFLINE") {
      detail.textContent = "Lần thử gần nhất không thấy board native";
    } else if (state === "CONNECTING") {
      detail.textContent = "Đang kiểm tra USB…";
    } else if (state === "ERROR") {
      detail.textContent = "Tác vụ lỗi · chưa có snapshot mới";
    } else {
      detail.textContent = "Chưa có snapshot board";
    }
  };

  const renderBoard = (board) => {
    if (!board) {
      text("#device-value", "Chưa có bằng chứng");
      text("#nvs-value", "—");
      text("#templates-value", "Chưa đọc");
      text("#fingerprint-value", "—");
      text("#device-contract", "USB native 303A:1001 · serial cố định · chưa có snapshot");
      text("#model-value", "—");
      text("#model-meta", "Pinned SHA-256 · chưa kiểm tra");
      text("#runtime-value", "—");
      text("#arena-value", "—");
      text("#board-state-chip", "—");
      return;
    }
    const device = board.device || {};
    text("#device-value", device.port ? `${device.port} · ${device.serial || "serial?"}` : "USB device");
    text("#device-contract", `USB ${(device.vid || 0).toString(16).padStart(4, "0").toUpperCase()}:${(device.pid || 0).toString(16).padStart(4, "0").toUpperCase()} · ${device.serial || "serial không có"}`);
    text("#nvs-value", board.persistence_state || "—");
    const labels = Array.isArray(board.labels) ? board.labels.join(" / ") : "nhãn chưa có";
    text("#templates-value", board.templates == null ? "Chưa đọc" : `${board.templates} mẫu · ${labels}`);
    text("#fingerprint-value", board.record_fingerprint || "—");
    text("#model-value", board.model_sha256 || "—");
    text("#model-meta", board.model_sha256 ? "SHA-256 board response · pinned contract" : "Board chưa trả model hash");
    text("#runtime-value", board.runtime_mode || "—");
    text("#arena-value", board.arena_used_bytes == null ? "—" : `${board.arena_used_bytes} / ${board.arena_capacity_bytes || "?"} B`);
    text("#board-state-chip", board.persistence_state || "—");
  };

  const phaseLabel = (phase) => ({
    QUEUED: "Đang chuẩn bị…",
    PREPARE: "Đọc file trên Mac…",
    CONNECTING: "Kết nối USB…",
    STATUS_READONLY: "Đọc mẫu trên board…",
    PARITY: "Kiểm tra model trên board…",
    MIC_PERMISSION: "Chờ quyền microphone · chưa thu âm",
    MIC_PREPARE: "Mở microphone · chưa thu âm",
    MIC_RECORDING: "Đang thu PCM — hãy nói",
    FRONTEND: "Chuẩn bị dữ liệu đối chứng…",
    PCM_TRANSFER: "PCM thô đã sẵn sàng…",
    BOARD_INFER: "Board đang nhận dạng…",
    QUERY: "Board đang nhận dạng file…",
    COMPLETE: "Hoàn tất",
    ERROR: "Không hoàn tất · xem lỗi bên dưới",
  }[phase] || "Đang xử lý…");

  const renderFlow = (job) => {
    const states = ["idle", "idle", "idle"];
    // This is a view of reported phases, not a timer or extra board probe.
    if (job && job.action !== "check_board") {
      if (job.status === "COMPLETED") states.fill("done");
      else if (job.status !== "ERROR") {
        if (["PREPARE", "MIC_PERMISSION", "MIC_PREPARE", "MIC_RECORDING", "FRONTEND", "PCM_TRANSFER"].includes(job.phase)) states[0] = "active";
        if (job.phase === "PARITY") states[1] = "active";
        if (["BOARD_INFER", "QUERY"].includes(job.phase)) { states[0] = "done"; states[1] = "active"; }
      }
    }
    ["audio", "board", "result"].forEach((name, index) => {
      const node = $(`#flow-${name}`);
      if (!node) return;
      node.dataset.state = states[index];
      node.querySelector(".step-state").textContent = {idle: "Chờ", active: "Đang chạy", done: "Xong"}[states[index]];
      if (states[index] === "active") node.setAttribute("aria-current", "step");
      else node.removeAttribute("aria-current");
    });
  };

  const renderProgress = (state) => {
    const job = state.active_job || state.last_job;
    const active = Boolean(state.active_job);
    const progress = job ? Number(job.progress || 0) : 0;
    $("#progress-bar").style.width = `${Math.max(0, Math.min(100, progress))}%`;
    text("#progress-percent", `${progress}%`);
    text("#progress-phase", job ? phaseLabel(job.phase) : "Chọn luồng PCM hoặc file để bắt đầu");
    renderFlow(job);
    text("#progress-message", job ? (job.message || "—") : "Một nút bấm sẽ tạo một job riêng trong artifacts/demo-dashboard.");
    text("#job-kind", active ? "RUNNING" : (job ? job.status : "IDLE"));
    const callout = $("#recording-callout");
    if (callout) callout.hidden = !(active && job.phase === "MIC_RECORDING");
    if (job && job.details) text("#recording-seconds", `${Number(job.details.recorded_seconds || 0).toFixed(1)} / 3s`);
    buttons.forEach((button) => { button.disabled = active; });
  };

  const scoreBar = (score) => {
    const value = Number(score);
    if (!Number.isFinite(value)) return 0;
    return Math.max(0, Math.min(100, ((value + 1) / 2) * 100));
  };

  const renderResultCard = (item, live = false) => {
    const correct = item.correct;
    const pillClass = correct === true ? "match-yes" : correct === false ? "match-no" : "match-unknown";
    const pill = correct === true ? "Đúng nhãn" : correct === false ? "Khác nhãn" : "Chưa đối chiếu";
    const scores = (item.scores || []).map((score) => `
      <div class="score-row">
        <span class="score-label">${html(score.label || "?")}</span>
        <div class="score-track"><div class="score-fill" style="width:${scoreBar(score.cosine)}%"></div></div>
        <span class="score-value">${html(Number(score.cosine).toFixed(4))}</span>
      </div>`).join("");
    const title = item.top1 || "—";
    const file = live ? "Thu PCM · 3 giây · board chọn" : `${item.file || "File kiểm tra"} · mẫu: ${item.expected_label || "chưa có"}`;
    const performance = item.performance || {};
    const capture = item.capture || {};
    const milliseconds = (value) => value == null || typeof value === "boolean" || !Number.isFinite(Number(value)) || Number(value) < 0 ? "—" : `${(Number(value) / 1000).toFixed(1)} ms`;
    const duration = capture.duration_seconds == null || typeof capture.duration_seconds === "boolean" || !Number.isFinite(Number(capture.duration_seconds)) || Number(capture.duration_seconds) < 0 ? "—" : `${Number(capture.duration_seconds).toFixed(3)} s`;
    const encoder = performance.encoder_us !== undefined ? performance.encoder_us
      : (performance.encoder_invoke_us !== undefined ? performance.encoder_invoke_us : item.encoder_invoke_us);
    const profile = capture.profile_id || "—";
    const device = capture.input_device && capture.input_device.name ? capture.input_device.name : "—";
    const acoustic = capture.acoustic_equivalence || "—";
    return `<article class="result-card">
      <div class="result-file">${html(file)}</div>
      <div class="result-head"><strong>${html(title)}</strong><span class="match-pill ${pillClass}">${html(pill)}</span></div>
      <p class="result-file">Cosine · mẫu trên board</p>
      <div class="score-list">${scores || "<span class=\"result-file\">Board không trả score</span>"}</div>
      ${live ? `<p class="result-file">Độ dài bản thu · ${html(duration)}</p>` : ""}
      <p class="latency">Xử lý trên board · ${html(milliseconds(performance.recognition_us))}</p>
      <p class="result-file">Tiền xử lý · ${html(milliseconds(performance.frontend_us))} · Encoder · ${html(milliseconds(encoder))}</p>
      <p class="result-file">Gửi USB + xử lý + nhận · ${html(milliseconds(performance.usb_roundtrip_us))}</p>
      ${live ? `<details><summary>Nguồn thu</summary>
      <p class="result-file">${html(device)} · 16 kHz · mono · PCM16</p>
      <p class="result-file">${html(profile)} · tương đương âm học INMP441: ${html(acoustic)}</p>
      <p class="result-file">Thu đủ 3 giây rồi gửi; tiền xử lý, model và cosine chạy trên board.</p></details>`
      : '<p class="result-file">Tensor đối chứng · bỏ qua tiền xử lý</p>'}
    </article>`;
  };

  const renderResults = (job) => {
    const empty = $("#results-empty");
    const list = $("#results-list");
    const summary = $("#results-summary");
    if (summary) { summary.hidden = true; summary.textContent = ""; }
    const result = job && job.result;
    if (!result || !list || !empty) { if (empty) empty.hidden = false; if (list) list.hidden = true; return; }
    let items = [];
    let live = false;
    if (Array.isArray(result.queries)) items = result.queries;
    else if (result.kind === "live_mac_query") { items = [result]; live = true; }
    if (!items.length) { empty.hidden = false; list.hidden = true; return; }
    empty.hidden = true;
    list.hidden = false;
    list.innerHTML = items.map((item) => renderResultCard(item, live)).join("");
    if (summary) {
      const labeled = items.filter((item) => typeof item.correct === "boolean");
      summary.textContent = live
        ? "Thu PCM raw Mac · chưa đối chiếu đúng/sai"
        : labeled.length === items.length
          ? `${labeled.filter((item) => item.correct).length}/${items.length} file đúng nhãn`
          : "Chưa đủ nhãn đối chiếu";
      summary.hidden = false;
    }
  };

  const renderHistory = (history) => {
    const node = $("#history-list");
    if (!node) return;
    if (!history || !history.length) { node.innerHTML = '<div class="empty-history">Chưa có job nào trong phiên này.</div>'; return; }
    node.innerHTML = history.map((job) => {
      const failed = job.status !== "COMPLETED";
      const detail = failed ? (job.error || "Lỗi không xác định") : (job.report_path || "report local");
      return `<div class="history-row">
        <div class="history-main"><strong>${html(job.label || job.action)}</strong><small>${html(escapeDate(job.started_at))} · ${html(job.id)}</small></div>
        <div class="history-main"><small>${html(detail)}</small></div>
        <div class="history-meta"><span class="history-status ${failed ? "error" : ""}">${failed ? "ERROR" : "COMPLETED"}</span><small>${html(escapeDate(job.finished_at))}</small></div>
      </div>`;
    }).join("");
  };

  const renderError = (state) => {
    const box = $("#error-box");
    if (!box) return;
    const error = state.last_error || (state.last_job && state.last_job.error);
    const message = state.connection === "OFFLINE" ? (error || OFFLINE_HINT) : error;
    box.hidden = !message;
    if (message) box.textContent = state.connection === "OFFLINE" ? message : `Không có fallback: ${message}`;
  };

  const render = (state) => {
    renderConnection(state.connection, state.last_check_at);
    renderBoard(state.board);
    renderProgress(state);
    renderResults(resultsSuppressed || state.active_job ? null : (state.last_job && state.last_job.status === "COMPLETED" ? state.last_job : null));
    renderHistory(state.history);
    renderError(state);
  };

  const refresh = async () => {
    try { render(await api("/api/state")); }
    catch (error) {
      resultsSuppressed = true;
      renderResults(null);
      $("#action-feedback").textContent = `Không đọc được state: ${error.message}`;
    }
  };

  const runAction = async (action) => {
    if (!csrfToken) return;
    resultsSuppressed = true;
    renderResults(null);
    buttons.forEach((button) => { button.disabled = true; });
    $("#action-feedback").textContent = `${buttonLabel(action)} đang khởi động…`;
    try {
      await api("/api/action", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": csrfToken },
        body: JSON.stringify({ action }),
      });
      resultsSuppressed = false;
      $("#action-feedback").textContent = "";
    } catch (error) {
      $("#action-feedback").textContent = error.message;
    } finally {
      await refresh();
    }
  };

  const buttonLabel = (action) => ({ check_board: "Kiểm tra board", run_frozen: "Query cố định", run_live: "Thu PCM" }[action] || action);
  buttons.forEach((button) => button.addEventListener("click", () => runAction(button.dataset.action)));

  const boot = async () => {
    try {
      const session = await api("/api/session");
      csrfToken = session.csrf_token;
      text("#manifest-value", session.prepared_manifest_sha256 || "—");
      await refresh();
      polling = window.setInterval(refresh, 700);
    } catch (error) {
      $("#action-feedback").textContent = `Dashboard chưa sẵn sàng: ${error.message}`;
    }
  };

  window.addEventListener("beforeunload", () => { if (polling) window.clearInterval(polling); });
  boot();
})();
