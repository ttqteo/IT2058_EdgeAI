"""Host-only guards for the local EdgeAI dashboard.

These tests inject a fake board client explicitly.  They do not establish
ESP32, USB, microphone, or browser behavior.
"""
from __future__ import annotations

import http.client
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import threading
import time
import wave
import zlib

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "demo_dashboard", Path(__file__).resolve().parents[1] / "scripts/demo_dashboard.py"
)
DASH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DASH)


def _raw_block(seed: int = 0, frames: int = 480) -> bytes:
    values = [((seed + index) % 65535) - 32768 for index in range(frames)]
    return struct.pack("<" + "h" * frames, *values)


class _RawStreamStop(Exception):
    pass


@pytest.mark.parametrize("overshoot", [False, True])
def test_raw_recorder_preserves_nonzero_bytes_and_rejects_overshoot(tmp_path, monkeypatch, overshoot):
    import sys
    import types

    class Stream:
        samplerate = 16_000
        channels = 1
        dtype = "int16"
        blocksize = 480
        active = True

        def __init__(self, **options):
            assert options["blocksize"] == DASH.AUDIO_BLOCKSIZE_FRAMES
            assert options["channels"] == 1
            assert options["dtype"] == "int16"
            assert options["dither_off"] is True
            assert options["clip_off"] is True
            self.callback = options["callback"]

        def start(self):
            try:
                for block in range(100):
                    frames = 480
                    if overshoot and block == 99:
                        frames = 481
                    self.callback(_raw_block(block, frames), frames, None, None)
            except _RawStreamStop:
                self.active = False

        def stop(self):
            self.active = False

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(
        RawInputStream=Stream,
        CallbackStop=_RawStreamStop,
        query_devices=lambda device, kind: {
            "index": 1,
            "name": "Synthetic input",
            "hostapi": "Synthetic",
            "max_input_channels": 1,
            "default_samplerate": 48_000,
        }))
    path = tmp_path / "raw.wav"
    if overshoot:
        with pytest.raises(DASH.DashboardError, match="not cropped"):
            DASH._record_mac_with_signal(path)
        assert not path.exists()
    else:
        metadata = DASH._record_mac_with_signal(path)
        expected = b"".join(_raw_block(block) for block in range(100))
        assert DASH.demo.read_pcm_capture(path) == expected
        assert metadata["provenance"] == DASH.RAW_INPUT_PROVENANCE
        assert metadata["input_device"]["name"] == "Synthetic input"
        assert metadata["requested_format"]["sample_rate_hz"] == 16_000
        assert metadata["effective_format"]["sample_rate_hz"] == 16_000
        assert metadata["sample_count"] == DASH.AUDIO_FRAME_COUNT
        assert metadata["pcm_bytes"] == 96_000


def test_raw_recorder_rejects_status_and_malformed_callback_bytes(tmp_path, monkeypatch):
    import sys
    import types

    class Stream:
        samplerate = 16_000
        channels = 1
        dtype = "int16"
        blocksize = 480
        active = True

        def __init__(self, **options):
            self.callback = options["callback"]

        def start(self):
            try:
                self.callback(_raw_block(), 480, None, "input_overflow")
            except _RawStreamStop:
                self.active = False

        def stop(self):
            self.active = False

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(
        RawInputStream=Stream,
        CallbackStop=_RawStreamStop,
        query_devices=lambda device, kind: {
            "index": 1, "name": "Synthetic input", "hostapi": "Synthetic",
            "max_input_channels": 1, "default_samplerate": 48_000,
        }))
    with pytest.raises(DASH.DashboardError, match="status"):
        DASH._record_mac_with_signal(tmp_path / "status.wav")

    class Malformed(Stream):
        def start(self):
            try:
                self.callback(_raw_block()[:-2], 480, None, None)
            except _RawStreamStop:
                self.active = False

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(
        RawInputStream=Malformed,
        CallbackStop=_RawStreamStop,
        query_devices=lambda device, kind: {
            "index": 1, "name": "Synthetic input", "hostapi": "Synthetic",
            "max_input_channels": 1, "default_samplerate": 48_000,
        }))
    with pytest.raises(DASH.DashboardError, match="byte length"):
        DASH._record_mac_with_signal(tmp_path / "malformed.wav")


def board_status(device=None):
    return {
        "protocol_version": 1,
        "model_sha256": DASH.demo.MODEL_SHA,
        "runtime_mode": "ESP_NN_ANSI_C_REFERENCE",
        "input_bytes": DASH.demo.INPUT_BYTES,
        "output_bytes": DASH.demo.OUTPUT_BYTES,
        "input_scale": DASH.demo.INPUT_SCALE,
        "input_zero_point": DASH.demo.INPUT_ZERO,
        "output_scale": DASH.demo.OUTPUT_SCALE,
        "output_zero_point": DASH.demo.OUTPUT_ZERO,
        "arena_capacity_bytes": 131072,
        "arena_used_bytes": 38140,
        "persistence_state": "READY",
        "persistence_source": "NVS_BOOT_READONLY",
        "enrollment_count": 2,
        "labels": ["Quang", "Tien"],
        "record_bytes": 1640,
        "record_crc32": 0xA00FF6A4,
        "record_fingerprint": "crc32:a00ff6a4",
        "device": device,
    }


class FakeBoard:
    def __init__(self, output: Path, *, top1="Quang", block: threading.Event | None = None):
        self.output = output
        self.device = {
            "port": "/dev/fake-serial",
            "serial": DASH.USB_SERIAL,
            "vid": DASH.USB_VID,
            "pid": DASH.USB_PID,
        }
        self.top1 = top1
        self.block = block
        self.closed = False
        self.recognize_calls = 0
        self.pcm_payloads = []

    def status(self):
        value = board_status()
        value.pop("device", None)
        return value

    def infer(self, payload):
        assert len(payload) == DASH.demo.INPUT_BYTES
        return bytes(DASH.demo.OUTPUT_BYTES), {
            "fixture_number": 1,
            "invoke_us": 100,
        }

    def recognize(self, payload):
        assert len(payload) == DASH.demo.INPUT_BYTES
        if self.block is not None:
            self.block.wait(timeout=3)
        self.recognize_calls += 1
        reply = {
            "top1": self.top1,
            "invoke_us": 53500,
        }
        scores = [
            {"label": self.top1, "cosine": 0.82},
            {"label": "Tien" if self.top1 == "Quang" else "Quang", "cosine": 0.63},
        ]
        return bytes(DASH.demo.OUTPUT_BYTES), scores, reply

    def recognize_pcm(self, payload):
        assert len(payload) == DASH.demo.PCM_BYTES
        self.pcm_payloads.append(bytes(payload))
        raw, scores, reply = self.recognize(bytes(DASH.demo.INPUT_BYTES))
        reply.update(operation="RECOGNIZE_PCM", frontend_us=1000)
        return raw, scores, reply

    def close(self):
        self.closed = True


def wait_idle(service: DASH.DashboardService, timeout=5):
    assert service.wait_for_idle(timeout), service.state()
    return service.state()["last_job"]


def test_service_initial_state_does_not_open_hardware_or_mic(tmp_path):
    calls = []

    def factory(_output):
        calls.append("hardware")
        raise AssertionError("hardware must be action-triggered")

    def recorder(*_args, **_kwargs):
        calls.append("mic")

    service = DASH.DashboardService(
        client_factory=factory,
        record_fn=recorder,
        output_root=tmp_path / "artifacts" / "demo-dashboard",
    )
    assert service.state()["connection"] == "NOT_CHECKED"
    assert service.state()["board"] is None
    assert calls == []
    assert not (tmp_path / "artifacts").exists()


def test_default_factory_is_real_usb_contract():
    service = DASH.DashboardService()
    assert service._client_factory == service._real_client_factory
    assert DASH.HOST == "127.0.0.1"
    assert DASH.PORT == 8765
    assert DASH.USB_VID == 0x303A
    assert DASH.USB_PID == 0x1001
    assert DASH.USB_SERIAL == "7C:0C:5F:5B:0E:00"


def test_ready_guard_accepts_both_declared_active_label_sets():
    DASH._require_ready({"persistence_state": "READY", "templates": 2,
                         "labels": ["Quang", "Tien"]})
    DASH._require_ready({"persistence_state": "READY", "templates": 3,
                         "labels": ["Quang", "Tien", "An"]})
    with pytest.raises(DASH.DashboardError):
        DASH._require_ready({"persistence_state": "READY", "templates": 3,
                             "labels": ["Quang", "Tien"]})


def test_status_job_reports_actual_device_nvs_and_unique_scoped_output(tmp_path):
    created = []

    def factory(output):
        client = FakeBoard(output)
        created.append(client)
        return client

    output_root = tmp_path / "artifacts" / "demo-dashboard"
    service = DASH.DashboardService(client_factory=factory, output_root=output_root)
    first = service.start_action("check_board")
    first_job = wait_idle(service)
    second = service.start_action("check_board")
    second_job = wait_idle(service)

    assert first["output_path"].startswith("artifacts/demo-dashboard/")
    assert second["output_path"].startswith("artifacts/demo-dashboard/")
    assert first["output_path"] != second["output_path"]
    assert first_job["status"] == "COMPLETED"
    assert first_job["result"]["persistence_state"] == "READY"
    assert service.state()["board"]["templates"] == 2
    assert service.state()["board"]["record_fingerprint"] == "crc32:a00ff6a4"
    assert service.state()["board"]["runtime_mode"] == "ESP_NN_ANSI_C_REFERENCE"
    assert service.state()["board"]["arena_used_bytes"] == 38140
    assert service.state()["last_check_at"]
    assert len(created) == 2
    assert all(item.closed for item in created)
    output_dirs = list(output_root.iterdir())
    assert len(output_dirs) == 2
    report = output_dirs[0] / "report.json"
    assert report.is_file()
    assert json.loads(report.read_text())["status"] == "COMPLETED"


@pytest.fixture
def synthetic_queries(monkeypatch, tmp_path):
    """Exercise job control with explicit synthetic tensors, never board evidence."""
    records = []
    for label in ("Quang", "Tien"):
        tensor = tmp_path / f"{label}.int8"
        tensor.write_bytes(bytes(DASH.demo.INPUT_BYTES))
        records.append({"role": "query", "label": label,
                        "source": {"path": f"{label}_test.m4a"},
                        "input": DASH.demo.binding(tensor)})
    monkeypatch.setattr(DASH.DashboardService, "_load_queries", lambda _self: records)
    return records


def test_frozen_job_marks_a_mismatch_false_and_keeps_encoder_latency(monkeypatch, tmp_path, synthetic_queries):
    monkeypatch.setattr(
        DASH.demo,
        "parity",
        lambda _client: [{"fixture_number": index, "max_absdiff_lsb": 0} for index in range(5)],
    )
    created = []

    def factory(output):
        client = FakeBoard(output, top1="Tien")
        created.append(client)
        return client

    service = DASH.DashboardService(
        client_factory=factory,
        output_root=tmp_path / "artifacts" / "demo-dashboard",
    )
    service.start_action("run_frozen")
    job = wait_idle(service)
    assert job["status"] == "COMPLETED", job
    assert job["result"]["query_count"] == 2
    assert job["result"]["closed_set_matches"] == 1
    assert job["result"]["queries"][0]["correct"] is False
    assert job["result"]["queries"][1]["correct"] is True
    assert job["result"]["queries"][0]["encoder_invoke_us"] == 53500
    assert job["result"]["accuracy_claim"] is False
    assert created[0].closed


def test_live_progress_separates_preparation_and_three_second_recording(tmp_path):
    phases = []
    timeline = []

    def record(destination, device=None, phase_callback=None):
        assert device is None
        assert callable(phase_callback)
        time.sleep(0.2)
        timeline.append("OPENED_SIGNAL")
        phase_callback("opened")
        time.sleep(0.2)
        timeline.append("CAPTURE_STARTED_SIGNAL")
        phase_callback("capture_started")
        with wave.open(str(destination), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(DASH.AUDIO_SAMPLE_RATE)
            handle.writeframes(np.zeros(DASH.AUDIO_FRAME_COUNT, dtype="<i2").tobytes())

    class RecordingService(DASH.DashboardService):
        def _set_progress(self, current_job_id, progress, phase, message, **details):
            phases.append(phase)
            timeline.append(f"PHASE:{phase}")
            super()._set_progress(current_job_id, progress, phase, message, **details)

    service = RecordingService(
        client_factory=lambda output: FakeBoard(output),
        record_fn=record,
        output_root=tmp_path / "artifacts" / "demo-dashboard",
    )
    # Synthetic capture only: require that no host conversion/DSP is invoked.
    original_convert = DASH.demo.convert_audio
    original_load = DASH.demo.load_wav
    original_mel = DASH.demo.log_mel
    original_quantize = DASH.demo.quantize
    original_embedding = DASH.demo.embedding

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Live raw PCM must not perform application DSP on Mac")

    try:
        DASH.demo.convert_audio = forbidden
        DASH.demo.load_wav = forbidden
        DASH.demo.log_mel = forbidden
        DASH.demo.quantize = forbidden
        DASH.demo.embedding = forbidden
        service.start_action("run_live")
        job = wait_idle(service)
    finally:
        DASH.demo.convert_audio = original_convert
        DASH.demo.load_wav = original_load
        DASH.demo.log_mel = original_mel
        DASH.demo.quantize = original_quantize
        DASH.demo.embedding = original_embedding

    assert job["status"] == "COMPLETED", job
    assert "MIC_PREPARE" in phases
    assert "MIC_RECORDING" in phases
    assert "BOARD_INFER" in phases
    assert timeline.index("CAPTURE_STARTED_SIGNAL") < timeline.index("PHASE:MIC_RECORDING")
    assert all(
        item != "PHASE:MIC_RECORDING"
        for item in timeline[: timeline.index("CAPTURE_STARTED_SIGNAL")]
    )
    assert job["result"]["correct"] is None
    assert job["result"]["performance"]["recognition_us"] is None


def test_injected_capture_is_unverified_and_report_binds_exact_pcm(tmp_path):
    raw = b"".join(_raw_block(block) for block in range(100))
    created = []

    def record(destination, device=None, phase_callback=None):
        assert device is None
        phase_callback("opened")
        phase_callback("capture_started")
        with wave.open(str(destination), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(DASH.AUDIO_SAMPLE_RATE)
            handle.writeframes(raw)
        # Synthetic recorder deliberately returns no metadata.
        return None

    def factory(output):
        client = FakeBoard(output)
        created.append(client)
        return client

    service = DASH.DashboardService(
        client_factory=factory,
        record_fn=record,
        output_root=tmp_path / "artifacts" / "demo-dashboard",
    )
    service.start_action("run_live")
    job = wait_idle(service)
    assert job["status"] == "COMPLETED", job
    report_path = next((tmp_path / "artifacts" / "demo-dashboard").glob("*/report.json"))
    report = json.loads(report_path.read_text())
    capture = report["recording"]
    assert capture["provenance"] == DASH.INJECTED_INPUT_PROVENANCE
    assert capture["provenance_verified"] is False
    assert capture["profile_id"] == "UNVERIFIED_INJECTED_RECORDER"
    assert capture["physical_mic"] is False
    assert capture["acoustic_equivalence"] == "UNVERIFIED"
    assert capture["pcm_bytes"] == len(raw) == 96_000
    assert capture["pcm_sha256"] == hashlib.sha256(raw).hexdigest()
    assert capture["pcm_crc32"] == zlib.crc32(raw) & 0xFFFFFFFF
    assert capture["wav_binding"]["sha256"]
    assert created[0].pcm_payloads == [raw]
    assert report["live"]["raw_pcm"] == {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "crc32": zlib.crc32(raw) & 0xFFFFFFFF,
    }


def test_input_device_numeric_selector_is_local_and_no_page_get_probe(tmp_path):
    assert DASH._parse_input_device("1") == 1
    assert DASH._parse_input_device(" MacBook Pro Microphone ") == "MacBook Pro Microphone"
    service = DASH.DashboardService(output_root=tmp_path / "artifacts" / "demo-dashboard")
    assert service.input_device is None
    assert service.state()["active_job"] is None


def test_error_is_visible_and_no_fake_result_or_fallback(tmp_path):
    def factory(_output):
        raise ConnectionError("native USB device unavailable")

    service = DASH.DashboardService(
        client_factory=factory,
        output_root=tmp_path / "artifacts" / "demo-dashboard",
    )
    service.start_action("check_board")
    job = wait_idle(service)
    state = service.state()
    assert job["status"] == "ERROR"
    assert state["connection"] == "OFFLINE"
    assert state["board"] is None
    assert job["result"] is None
    assert "fallback" not in (job["error"] or "").lower()
    assert "USB native" in (job["error"] or "")
    assert "không tự reset" in (job["error"] or "")


def test_missing_native_usb_value_error_is_offline_with_friendly_hint(tmp_path):
    def factory(_output):
        raise ValueError("Need exactly one matching ESP32 USB device")

    service = DASH.DashboardService(
        client_factory=factory,
        output_root=tmp_path / "artifacts" / "demo-dashboard",
    )
    service.start_action("check_board")
    job = wait_idle(service)
    state = service.state()
    assert job["status"] == "ERROR"
    assert state["connection"] == "OFFLINE"
    assert "303A:1001" in (job["error"] or "")
    assert DASH.USB_SERIAL in (job["error"] or "")
    assert "không tự reset" in (job["error"] or "")
    assert "Need exactly" not in (job["error"] or "")


def test_report_write_failure_clears_completed_result(monkeypatch, tmp_path):
    service = DASH.DashboardService(
        client_factory=lambda output: FakeBoard(output),
        output_root=tmp_path / "artifacts" / "demo-dashboard",
    )

    def fail_report(*_args, **_kwargs):
        raise OSError("report disk full")

    monkeypatch.setattr(service, "_write_report", fail_report)
    service.start_action("check_board")
    job = wait_idle(service)
    assert job["status"] == "ERROR"
    assert job["report_path"] is None
    assert job["result"] is None
    assert "report disk full" in (job["error"] or "")
    assert service.state()["active_job"] is None


def test_dashboard_sources_guard_stale_results_and_mobile_implicit_width():
    web_root = Path(__file__).resolve().parents[1] / "scripts" / "demo_dashboard_web"
    app_source = (web_root / "app.js").read_text(encoding="utf-8")
    css_source = (web_root / "style.css").read_text(encoding="utf-8")
    assert 'state.last_job.status === "COMPLETED"' in app_source
    assert 'state.active_job ? null' in app_source
    assert 'ONLINE: ["Đã đọc board"' in app_source
    assert "Lần đọc gần nhất" in app_source
    assert "USB native ESP32-S3" in app_source
    assert "board.labels" in app_source
    assert "grid-template-columns: minmax(0, 1fr) minmax(0, 1fr)" in css_source
    assert ".evidence-grid > *" in css_source
    assert "min-width: 0" in css_source
    assert "overflow-wrap: anywhere" in css_source


def test_pinned_model_error_finishes_job_instead_of_leaving_it_active(monkeypatch, tmp_path):
    monkeypatch.setattr(DASH.demo, "MODEL", tmp_path / "missing-model.tflite")
    service = DASH.DashboardService(
        client_factory=lambda _output: (_ for _ in ()).throw(AssertionError("no hardware")),
        output_root=tmp_path / "artifacts" / "demo-dashboard",
    )
    service.start_action("check_board")
    job = wait_idle(service)
    assert job["status"] == "ERROR"
    assert service.state()["active_job"] is None
    assert "Pinned model" in (job["error"] or "")


def test_single_active_job_rejects_concurrent_action(tmp_path, synthetic_queries):
    release = threading.Event()

    def factory(output):
        return FakeBoard(output, block=release)

    service = DASH.DashboardService(
        client_factory=factory,
        output_root=tmp_path / "artifacts" / "demo-dashboard",
    )
    # Start the frozen job and wait until the fake recognize call owns the job.
    monkey = pytest.MonkeyPatch()
    monkey.setattr(DASH.demo, "parity", lambda _client: [])
    try:
        service.start_action("run_frozen")
        deadline = time.monotonic() + 2
        while service.state()["active_job"] is None or service.state()["active_job"]["phase"] != "QUERY":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        with pytest.raises(DASH.BusyError):
            service.start_action("check_board")
    finally:
        release.set()
        monkey.undo()
    assert wait_idle(service)["status"] == "COMPLETED"


def request(server, method, path, *, headers=None, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    return response.status, payload


@pytest.fixture
def running_server(tmp_path):
    def unavailable(_output):
        raise ConnectionError("native USB device unavailable")

    service = DASH.DashboardService(
        client_factory=unavailable,
        output_root=tmp_path / "artifacts" / "demo-dashboard",
    )
    server = DASH.make_server(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, service
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_http_routes_are_loopback_csrf_bounded_and_non_exposing(running_server):
    server, service = running_server
    host = f"127.0.0.1:{server.server_port}"
    status, body = request(server, "GET", "/api/session", headers={"Host": host})
    assert status == 200
    session = json.loads(body)
    assert session["csrf_token"] == service.csrf_token
    assert session["port"] == server.server_port

    status, _ = request(server, "GET", "/api/state", headers={"Host": host})
    assert status == 200
    for path in ("/performance", "/performance.js", "/performance.css"):
        status, payload = request(server, "GET", path, headers={"Host": host})
        assert status == 200
        assert payload
    assert service.state()["active_job"] is None
    status, _ = request(server, "GET", "/scripts/demo_voice_usb.py", headers={"Host": host})
    assert status == 404
    status, _ = request(server, "GET", "/../scripts/demo_voice_usb.py", headers={"Host": host})
    assert status == 404
    status, _ = request(server, "GET", "/api/state", headers={"Host": "evil.example"})
    assert status == 400

    headers = {
        "Host": host,
        "Origin": f"http://127.0.0.1:{server.server_port}",
        "Content-Type": "application/json",
        "X-CSRF-Token": service.csrf_token,
    }
    status, _ = request(server, "POST", "/api/action", headers=headers, body=json.dumps({"action": "provision"}))
    assert status == 400
    status, _ = request(server, "POST", "/api/action", headers={**headers, "X-CSRF-Token": "wrong"}, body='{"action":"check_board"}')
    assert status == 403
    status, _ = request(server, "POST", "/api/action", headers={**headers, "Origin": "https://evil.example"}, body='{"action":"check_board"}')
    assert status == 403
    status, _ = request(server, "POST", "/api/action", headers=headers, body="x" * (DASH.MAX_JSON_BODY + 1))
    assert status in {400, 413}


def test_http_action_requires_valid_origin_and_returns_one_job(running_server):
    server, service = running_server
    host = f"127.0.0.1:{server.server_port}"
    session_status, session_body = request(server, "GET", "/api/session", headers={"Host": host})
    assert session_status == 200
    token = json.loads(session_body)["csrf_token"]
    headers = {
        "Host": host,
        "Origin": f"http://127.0.0.1:{server.server_port}",
        "Content-Type": "application/json",
        "X-CSRF-Token": token,
    }
    # Default service is not called unless this explicit action is sent; an
    # unavailable-board error is acceptable and must remain visible in state.
    status, payload = request(server, "POST", "/api/action", headers=headers, body='{"action":"check_board"}')
    assert status == 202
    assert json.loads(payload)["job"]["action"] == "check_board"
    assert service.wait_for_idle(3)
    assert service.state()["last_job"]["status"] == "ERROR"
