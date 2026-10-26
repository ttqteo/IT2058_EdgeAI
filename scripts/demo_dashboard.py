#!/usr/bin/env python3
"""Loopback-only monitor for the EdgeAI two-speaker board demonstration.

The dashboard is intentionally a thin UI adapter around the frozen USB demo
client.  It never provisions, resets, flashes, erases, or polls hardware in
the background.  The default backend is always the real USB client; tests may
inject a fake client explicitly.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import secrets
import sys
import threading
import time
import wave
import zlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
WEB_ROOT = SCRIPTS / "demo_dashboard_web"
ARTIFACT_ROOT = ROOT / "artifacts" / "demo-dashboard"
HOST = "127.0.0.1"
PORT = 8765
USB_VID = 0x303A
USB_PID = 0x1001
USB_SERIAL = "7C:0C:5F:5B:0E:00"
MAX_JSON_BODY = 32 * 1024
MAX_HISTORY = 20
AUDIO_SAMPLE_RATE = 16_000
AUDIO_DURATION_SECONDS = 3
AUDIO_CHANNELS = 1
AUDIO_FRAME_COUNT = AUDIO_SAMPLE_RATE * AUDIO_DURATION_SECONDS
AUDIO_SAMPLE_WIDTH_BYTES = 2
AUDIO_BLOCKSIZE_FRAMES = 480
RAW_INPUT_PROFILE_ID = "MAC_RAW_PCM16_MONO16K_V1"
RAW_INPUT_PROVENANCE = "NATIVE_SOUNDDEVICE_RAWINPUTSTREAM"
INJECTED_INPUT_PROVENANCE = "INJECTED_RECORDER_UNVERIFIED"
ACOUSTIC_EQUIVALENCE = "UNVERIFIED"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/performance": ("performance.html", "text/html; charset=utf-8"),
    "/performance.js": ("performance.js", "text/javascript; charset=utf-8"),
    "/performance.css": ("performance.css", "text/css; charset=utf-8"),
}
ACTION_NAMES = {"check_board", "run_frozen", "run_live"}
ACTION_LABELS = {
    "check_board": "Kiểm tra board",
    "run_frozen": "Chạy 2 query cố định",
    "run_live": "Thu PCM · 3 giây",
}
ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])/(?:[^\s'\"]+/)+[^\s'\"]*")
OFFLINE_DEVICE_MESSAGE = (
    "Không tìm thấy đúng USB native ESP32-S3 "
    f"(VID/PID {USB_VID:04X}:{USB_PID:04X}, serial {USB_SERIAL}). "
    "Hãy kết nối đúng board; dashboard không tự reset, erase hoặc flash."
)


def _load_demo_client():
    """Import the existing client without changing that file's source."""

    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return importlib.import_module("demo_voice_usb")


demo = _load_demo_client()


class DashboardError(RuntimeError):
    """Expected, user-facing dashboard failure."""


class BusyError(DashboardError):
    """A second hardware action arrived while one job owns the serial lock."""


class RequestError(DashboardError):
    """A request failed a local HTTP/API guard."""


class PayloadLimitError(RequestError):
    """The bounded JSON request body is too large."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _redact_text(value: Any) -> str:
    text = str(value).strip() or value.__class__.__name__
    text = text.replace(str(ROOT), "<workspace>")
    text = text.replace(str(ARTIFACT_ROOT), "<dashboard-output>")
    text = ABSOLUTE_PATH_RE.sub("<local-path>", text)
    return text[:600]


def _is_offline_error(error: BaseException) -> bool:
    """Recognize a missing/unavailable native USB device without broad fallback."""

    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    error_name = type(error).__name__.lower()
    if error_name in {"usbvalueerror", "usbconnectionerror"}:
        return True
    message = str(error).lower()
    if isinstance(error, OSError) and any(
        marker in message for marker in ("usb", "serial", "open port", "no such device")
    ):
        return True
    return (
        "need exactly one matching esp32 usb device" in message
        or "no matching usb device" in message
        or "native usb device" in message
    )


def _user_error(error: BaseException, *, allow_offline_hint: bool = True) -> str:
    if allow_offline_hint and _is_offline_error(error):
        return OFFLINE_DEVICE_MESSAGE
    return _redact_text(error)


def _parse_input_device(device: str | int | None) -> str | int | None:
    """Normalize the local sounddevice selector without touching the system default."""

    if device is None or (isinstance(device, int) and not isinstance(device, bool)):
        return device
    if not isinstance(device, str):
        raise DashboardError("Input device must be a device name or numeric index")
    value = device.strip()
    if not value:
        raise DashboardError("Input device must not be empty")
    if value.isdecimal():
        return int(value)
    return value


def _device_display(device: str | int | None) -> str:
    return "system-default" if device is None else str(device)


def _input_device_info(sd: Any, requested: str | int | None) -> tuple[dict[str, Any], int | str | None]:
    """Resolve the selected input without opening a stream or changing global defaults."""

    query_devices = getattr(sd, "query_devices", None)
    if not callable(query_devices):
        raise DashboardError("sounddevice cannot enumerate the selected input device")
    info = query_devices(requested, "input")
    if not hasattr(info, "get"):
        raise DashboardError("sounddevice returned malformed input-device metadata")
    max_channels = info.get("max_input_channels")
    if type(max_channels) not in (int, float) or not math.isfinite(float(max_channels)):
        raise DashboardError("Input device did not report max_input_channels")
    if max_channels < AUDIO_CHANNELS:
        raise DashboardError("Selected input device has no input channel")
    name = info.get("name")
    if not isinstance(name, str) or not name.strip():
        raise DashboardError("Input device did not report a name")
    index = info.get("index")
    if type(index) is not int or index < 0:
        index = requested
    hostapi = info.get("hostapi")
    if type(hostapi) not in (int, str) or isinstance(hostapi, bool):
        hostapi = None
    default_rate = info.get("default_samplerate")
    if type(default_rate) not in (int, float) or isinstance(default_rate, bool) or not math.isfinite(float(default_rate)):
        default_rate = None
    public = {
        "requested": _device_display(requested),
        "index": index,
        "name": name,
        "hostapi": hostapi,
        "max_input_channels": int(max_channels),
        "default_samplerate_hz": default_rate,
    }
    # Passing the resolved index makes the report and the opened stream refer
    # to the same device.  None remains a valid fallback only when a test or
    # backend cannot expose an index.
    return public, index


def _format_rate(value: Any) -> int | float | None:
    if type(value) not in (int, float) or isinstance(value, bool):
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        return None
    return int(value) if value.is_integer() else value


def _stream_effective_format(stream: Any) -> dict[str, Any]:
    """Read stream properties when the backend exposes them; never fill them by inference."""

    channels = getattr(stream, "channels", None)
    if type(channels) is not int or channels < 0:
        channels = None
    blocksize = getattr(stream, "blocksize", None)
    if type(blocksize) is not int or blocksize < 0:
        blocksize = None
    dtype = getattr(stream, "dtype", None)
    if dtype is not None:
        dtype = str(dtype)
    return {
        "sample_rate_hz": _format_rate(getattr(stream, "samplerate", None)),
        "channels": channels,
        "sample_format": dtype,
        "blocksize_frames": blocksize,
    }


def _validate_effective_format(effective: dict[str, Any]) -> None:
    """Reject a reported stream format that cannot be labeled as the PCM profile."""

    rate = effective.get("sample_rate_hz")
    if rate is not None and rate != AUDIO_SAMPLE_RATE:
        raise DashboardError(
            f"Raw input effective samplerate is {rate} Hz; refusing to relabel it as 16 kHz"
        )
    channels = effective.get("channels")
    if channels is not None and channels != AUDIO_CHANNELS:
        raise DashboardError("Raw input effective channel count is not mono")
    sample_format = effective.get("sample_format")
    if sample_format is not None and sample_format.lower() not in {"int16", "<i2", "pcm16"}:
        raise DashboardError("Raw input effective sample format is not int16")
    blocksize = effective.get("blocksize_frames")
    if blocksize is not None and blocksize != AUDIO_BLOCKSIZE_FRAMES:
        raise DashboardError("Raw input effective blocksize is not 480 frames")


def _pcm_binding(wav_path: Path, payload: bytes) -> dict[str, Any]:
    """Bind both the WAV container and the exact PCM payload sent to the board."""

    crc32 = zlib.crc32(payload) & 0xFFFFFFFF
    return {
        "wav_binding": demo.binding(wav_path),
        "pcm_bytes": len(payload),
        "pcm_sha256": hashlib.sha256(payload).hexdigest(),
        "pcm_crc32": crc32,
        "pcm_crc32_hex": f"{crc32:08x}",
    }


def _record_mac_with_signal(
    destination: Path,
    device: str | int | None = None,
    *,
    phase_callback: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Record exactly three seconds and signal actual stream lifecycle events.

    This dashboard-local recorder deliberately does not use ``demo.record_mac``:
    that helper has a presentation delay before ``sd.rec`` and cannot tell the
    dashboard when the capture stream has really started.  ``RawInputStream``
    supplies a bytes-like callback buffer; the callback stores those bytes as-is
    and rejects status, frame-count, and byte-count violations.  No NumPy audio
    conversion, crop, padding, resampling, denoise, gain, or AGC happens here.
    """

    import sounddevice as sd

    if destination.exists():
        raise DashboardError("Refusing to overwrite an existing raw capture")
    requested_device = _parse_input_device(device)
    input_device, stream_device = _input_device_info(sd, requested_device)
    requested_format = {
        "sample_rate_hz": AUDIO_SAMPLE_RATE,
        "channels": AUDIO_CHANNELS,
        "sample_format": "int16",
        "endianness": "little",
        "sample_width_bytes": AUDIO_SAMPLE_WIDTH_BYTES,
        "blocksize_frames": AUDIO_BLOCKSIZE_FRAMES,
    }
    chunks: list[bytes] = []
    captured_frames = 0
    captured_bytes = 0
    callback_count = 0
    capture_done = threading.Event()
    capture_errors: list[str] = []

    def signal(event: str) -> None:
        if phase_callback is not None:
            phase_callback(event)

    def reject(message: str) -> None:
        capture_errors.append(message)
        capture_done.set()
        raise sd.CallbackStop

    def callback(indata: Any, frames: int, _time_info: Any, status: Any) -> None:
        nonlocal captured_frames, captured_bytes, callback_count
        if status:
            reject("Raw microphone callback reported a stream status; capture rejected")
        if type(frames) is not int or frames <= 0:
            reject("Raw microphone callback returned malformed frame count")
        try:
            raw = bytes(indata)
        except (TypeError, ValueError) as error:
            reject(f"Raw microphone callback returned a non-bytes buffer: {error}")
        expected_bytes = frames * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH_BYTES
        if len(raw) != expected_bytes:
            reject("Raw microphone callback byte length does not match frame count")
        if captured_frames + frames > AUDIO_FRAME_COUNT:
            reject("Raw microphone callback overshot the capture window; raw PCM was not cropped")
        chunks.append(raw)
        captured_frames += frames
        captured_bytes += len(raw)
        callback_count += 1
        if captured_frames == AUDIO_FRAME_COUNT:
            capture_done.set()
            raise sd.CallbackStop

    stream = sd.RawInputStream(
        samplerate=AUDIO_SAMPLE_RATE,
        channels=AUDIO_CHANNELS,
        dtype="int16",
        device=stream_device,
        callback=callback,
        blocksize=AUDIO_BLOCKSIZE_FRAMES,
        # These are PortAudio requests only; CoreAudio/device DSP remains
        # unknown and is disclosed in the returned provenance metadata.
        dither_off=True,
        clip_off=True,
    )
    signal("opened")
    started = False
    effective_format: dict[str, Any] = {
        "sample_rate_hz": None,
        "channels": None,
        "sample_format": None,
        "blocksize_frames": None,
    }
    try:
        stream.start()
        started = True
        effective_format = _stream_effective_format(stream)
        effective_format["endianness"] = "little"
        effective_format["sample_width_bytes"] = AUDIO_SAMPLE_WIDTH_BYTES
        _validate_effective_format(effective_format)
        signal("capture_started")
        while not capture_done.wait(0.05):
            if getattr(stream, "active", True) is False:
                raise DashboardError("Raw microphone stream stopped before 3 seconds")
    finally:
        if started:
            stream.stop()
        stream.close()

    if capture_errors:
        raise DashboardError(capture_errors[0])
    if not chunks or captured_frames != AUDIO_FRAME_COUNT:
        raise DashboardError("Raw microphone did not return exactly 48000 samples")
    payload = b"".join(chunks)
    if captured_bytes != AUDIO_FRAME_COUNT * AUDIO_SAMPLE_WIDTH_BYTES or len(payload) != captured_bytes:
        raise DashboardError("Raw microphone did not return exactly 96000 PCM bytes")
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(AUDIO_CHANNELS)
        handle.setsampwidth(AUDIO_SAMPLE_WIDTH_BYTES)
        handle.setframerate(AUDIO_SAMPLE_RATE)
        handle.writeframes(payload)
    return {
        "profile_id": RAW_INPUT_PROFILE_ID,
        "provenance": RAW_INPUT_PROVENANCE,
        "provenance_verified": True,
        "input_device": input_device,
        "requested_format": requested_format,
        "effective_format": effective_format,
        "sample_count": captured_frames,
        "pcm_bytes": captured_bytes,
        "callback_count": callback_count,
        "physical_mic": False,
        "acoustic_equivalence": ACOUSTIC_EQUIVALENCE,
        "acoustic_equivalence_note": "Not calibrated or claimed equivalent to INMP441",
        "os_device_dsp": "UNKNOWN",
        "coreaudio_conversion": "POSSIBLE",
        "portaudio_requests": {"dither_off": True, "clip_off": True},
        "transport_mode": "BATCH_CAPTURE_THEN_RECOGNIZE_PCM",
    }


def _json_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _public_device(device: Any) -> Optional[dict[str, Any]]:
    if not isinstance(device, dict):
        return None
    result: dict[str, Any] = {}
    for key in ("port", "serial", "vid", "pid", "backend"):
        if key in device:
            value = device[key]
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                result[key] = value
    return result or None


def _artifact_summary(path: Path, expected_sha: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise DashboardError("Pinned model file is unavailable")
    raw = path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise DashboardError("Pinned model hash mismatch")
    return {"name": path.name, "bytes": len(raw), "sha256": actual_sha}


def _board_summary(client: Any, reply: dict[str, Any]) -> dict[str, Any]:
    """Keep only board evidence needed by the UI/report."""

    if not isinstance(reply, dict):
        raise DashboardError("Board status was not a JSON object")
    state = reply.get("persistence_state")
    count = reply.get("enrollment_count")
    record_crc = reply.get("record_crc32")
    fingerprint = reply.get("record_fingerprint")
    summary = {
        "device": _public_device(getattr(client, "device", None)),
        "persistence_state": state if isinstance(state, str) else None,
        "templates": count if isinstance(count, int) and not isinstance(count, bool) else None,
        "labels": reply.get("labels") if isinstance(reply.get("labels"), list) and
            all(isinstance(item, str) for item in reply["labels"]) else None,
        "record_fingerprint": fingerprint if isinstance(fingerprint, str) else None,
        "record_crc32": record_crc if isinstance(record_crc, int) and not isinstance(record_crc, bool) else None,
        "record_bytes": reply.get("record_bytes") if isinstance(reply.get("record_bytes"), int) else None,
        "model_sha256": reply.get("model_sha256") if isinstance(reply.get("model_sha256"), str) else None,
        "runtime_mode": reply.get("runtime_mode") if isinstance(reply.get("runtime_mode"), str) else None,
        "arena_used_bytes": reply.get("arena_used_bytes") if isinstance(reply.get("arena_used_bytes"), int) else None,
        "arena_capacity_bytes": reply.get("arena_capacity_bytes") if isinstance(reply.get("arena_capacity_bytes"), int) else None,
        "telemetry": demo.validate_telemetry(reply),
    }
    return summary


def _query_performance(reply: dict[str, Any], roundtrip_us: int) -> dict[str, Any]:
    """Keep measured board memory separate from host USB request timing."""
    telemetry = demo.validate_telemetry(reply)
    encoder_us = reply.get("invoke_us")
    if type(encoder_us) is not int or encoder_us < 0:
        encoder_us = None
    frontend_us = reply.get("frontend_us")
    if type(frontend_us) is not int or frontend_us < 0:
        frontend_us = None
    if type(roundtrip_us) is not int or roundtrip_us < 0:
        roundtrip_us = None
    return {
        "request_id": reply.get("request_id"),
        "encoder_us": encoder_us,
        # Retain the old key for report consumers; both keys name the same
        # board Invoke measurement and are never synthesized from another one.
        "encoder_invoke_us": encoder_us,
        "frontend_us": frontend_us,
        "recognition_us": telemetry.get("recognition_us") if telemetry else None,
        "operation": reply.get("operation"),
        "pcm_bytes": demo.PCM_BYTES if reply.get("operation") == "RECOGNIZE_PCM" else None,
        "usb_roundtrip_us": roundtrip_us,
        "input_bytes": reply.get("input_bytes"),
        "output_bytes": reply.get("output_bytes"),
        "arena_used_bytes": reply.get("arena_used_bytes"),
        "arena_capacity_bytes": reply.get("arena_capacity_bytes"),
        "telemetry": telemetry,
        "timing_scope": (
            "BOARD_PCM_FRONTEND_ENCODER_COSINE"
            if reply.get("operation") == "RECOGNIZE_PCM"
            else "LEGACY_TENSOR_EXCLUDES_FRONTEND"
        ),
    }


def _performance_summary(board: Any, checked_at: Any, history: list[dict[str, Any]]) -> dict[str, Any]:
    """Only committed successful jobs supply last-query evidence; never probe USB."""
    last_query = None
    for job in history:
        if job.get("status") != "COMPLETED":
            continue
        result = job.get("result") or {}
        queries = result.get("queries") or ([result] if result.get("kind") == "live_mac_query" else [])
        if queries:
            query = queries[-1]
            last_query = {
                "job_id": job["id"], "sampled_at": job.get("finished_at"),
                "source": query.get("file") or "Thu PCM · 3 giây",
                "top1": query.get("top1"), "metrics": copy.deepcopy(query.get("performance")),
            }
            break
    memory = copy.deepcopy(board.get("telemetry")) if board else None
    memory_at = checked_at if memory is not None else None
    if last_query and last_query["metrics"]:
        latest = last_query["metrics"].get("telemetry")
        if latest is not None and (memory_at is None or (last_query["sampled_at"] or "") >= memory_at):
            memory, memory_at = copy.deepcopy(latest), last_query["sampled_at"]
    return {"memory_snapshot": memory, "sampled_at": memory_at, "last_query": last_query}


def _require_ready(board: dict[str, Any]) -> None:
    count = board.get("templates")
    labels = board.get("labels")
    if (board.get("persistence_state") != "READY" or not isinstance(count, int)
            or not 2 <= count <= demo.MAX_TEMPLATES
            or not demo._valid_active_labels(labels, count)):
        raise DashboardError(
            "Board NVS chưa READY với tập nhãn đang được hỗ trợ; không gửi query"
        )


def _safe_relative(path: Path, root: Path = ARTIFACT_ROOT) -> str:
    """Return a report path only if it stays under the dashboard artifact root."""

    root = root.resolve()
    candidate = path.resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise DashboardError("Dashboard output escaped its artifact directory") from error
    return str(Path("artifacts") / "demo-dashboard" / relative)


class DashboardService:
    """State machine for one local dashboard process.

    ``client_factory`` and ``record_fn`` are deliberately injectable for unit
    tests.  The production default is the unchanged real USB client and the
    dashboard-local recorder with an actual stream-start callback.
    """

    def __init__(
        self,
        *,
        root: Path = ROOT,
        client_factory: Optional[Callable[[Path], Any]] = None,
        record_fn: Optional[Callable[..., Any]] = None,
        input_device: str | int | None = None,
        output_root: Optional[Path] = None,
    ) -> None:
        self.root = root.resolve()
        self.output_root = (output_root or self.root / "artifacts" / "demo-dashboard").resolve()
        self.port = PORT
        self._client_factory = client_factory or self._real_client_factory
        self._record_fn = record_fn or _record_mac_with_signal
        self._native_recorder = record_fn is None
        self.input_device = _parse_input_device(input_device)
        self._lock = threading.RLock()
        self._serial_lock = threading.Lock()
        self._csrf_token = secrets.token_urlsafe(32)
        self._active: Optional[dict[str, Any]] = None
        self._last_job: Optional[dict[str, Any]] = None
        self._history: list[dict[str, Any]] = []
        self._board: Optional[dict[str, Any]] = None
        self._connection = "NOT_CHECKED"
        self._last_check_at: Optional[str] = None
        self._last_error: Optional[str] = None

    def _public_path(self, path: Path) -> str:
        return _safe_relative(path, self.output_root)

    @property
    def csrf_token(self) -> str:
        return self._csrf_token

    def _real_client_factory(self, output: Path) -> Any:
        # USBEncoder performs the exact VID/PID and serial-number match.
        return demo.USBEncoder(USB_SERIAL, output)

    def session(self) -> dict[str, Any]:
        return {
            "csrf_token": self._csrf_token,
            "host": HOST,
            "port": self.port,
            "actions": sorted(ACTION_NAMES),
            "model_sha256": demo.MODEL_SHA,
            "prepared_manifest_sha256": demo.VOICE_MANIFEST_SHA,
        }

    def _state_job(self, job: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        return copy.deepcopy(job) if job is not None else None

    def state(self) -> dict[str, Any]:
        with self._lock:
            active = self._state_job(self._active)
            last = self._state_job(self._last_job)
            return {
                "server": {"host": HOST, "port": self.port},
                "connection": self._connection,
                "last_check_at": self._last_check_at,
                "board": copy.deepcopy(self._board),
                "active_job": active,
                "last_job": last,
                "history": copy.deepcopy(self._history),
                "last_error": self._last_error,
                "performance": _performance_summary(self._board, self._last_check_at, self._history),
            }

    def _set_connection(self, value: str) -> None:
        with self._lock:
            self._connection = value

    def _set_board(self, board: dict[str, Any]) -> None:
        with self._lock:
            self._board = copy.deepcopy(board)
            self._connection = "ONLINE"
            self._last_check_at = _utc_now()
            self._last_error = None

    def _set_progress(self, job_id: str, progress: int, phase: str, message: str, **details: Any) -> None:
        with self._lock:
            if self._active is None or self._active["id"] != job_id:
                return
            self._active["progress"] = max(0, min(100, int(progress)))
            self._active["phase"] = phase
            self._active["message"] = message
            if details:
                self._active.setdefault("details", {}).update(details)

    def _make_output_dir(self, job_id: str) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for _ in range(8):
            path = self.output_root / f"{job_id}-{secrets.token_hex(4)}"
            try:
                path.mkdir(mode=0o700)
                return path
            except FileExistsError:
                continue
        raise DashboardError("Could not allocate a unique dashboard output directory")

    def start_action(self, action: str) -> dict[str, Any]:
        if action not in ACTION_NAMES:
            raise RequestError("Unknown dashboard action")
        with self._lock:
            if self._active is not None:
                raise BusyError("Một tác vụ board đang chạy; hãy chờ hoàn tất")
            job_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(4)
            output = self._make_output_dir(job_id)
            job = {
                "id": job_id,
                "action": action,
                "label": ACTION_LABELS[action],
                "status": "RUNNING",
                "progress": 0,
                "phase": "QUEUED",
                "message": "Đã xếp hàng; chưa gọi hardware",
                "started_at": _utc_now(),
                "finished_at": None,
                "output_path": self._public_path(output),
                "report_path": None,
                "details": {},
                "result": None,
                "error": None,
            }
            self._active = job
            self._board = None
            self._connection = "CONNECTING"
            self._last_error = None
            worker = threading.Thread(
                target=self._run_job,
                args=(job_id, action, output),
                name=f"edgeai-dashboard-{job_id}",
                daemon=True,
            )
            worker.start()
            return copy.deepcopy(job)

    def _write_report(self, output: Path, report: dict[str, Any]) -> str:
        path = output / "report.json"
        with path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        return self._public_path(path)

    def _base_report(self, job_id: str, action: str, output: Path) -> dict[str, Any]:
        return {
            "scope": "LOCAL_EDGEAI_DASHBOARD_DEMO_NOT_ACCURACY_BENCHMARK",
            "job_id": job_id,
            "action": action,
            "status": "RUNNING",
            "backend": "usb",
            "usb_contract": {
                "vid": USB_VID,
                "pid": USB_PID,
                "serial": USB_SERIAL,
                "host": HOST,
            },
            "model": {
                **_artifact_summary(Path(demo.MODEL), demo.MODEL_SHA),
            },
            "prepared_manifest_sha256": demo.VOICE_MANIFEST_SHA,
            "output_scope": self._public_path(output),
            "nvs_policy": "READ_ONLY_DURING_DASHBOARD_ACTIONS_NO_PROVISION_RESET_ERASE_FLASH",
            "privacy": "Mac recordings stay in this local job directory and are never uploaded",
        }

    @staticmethod
    def _summary_result(board: dict[str, Any]) -> dict[str, Any]:
        return {
            "kind": "board_status",
            "persistence_state": board.get("persistence_state"),
            "templates": board.get("templates"),
            "record_fingerprint": board.get("record_fingerprint"),
            "runtime_mode": board.get("runtime_mode"),
        }

    def _run_status(self, job_id: str, output: Path, report: dict[str, Any]) -> None:
        self._set_progress(job_id, 15, "CONNECTING", "Đang kết nối đúng USB Serial/JTAG…")
        client = self._client_factory(output)
        try:
            self._set_progress(job_id, 55, "STATUS_READONLY", "Đọc STATUS read-only; không thay đổi NVS…")
            status = client.status()
            board = _board_summary(client, status)
            self._set_board(board)
            report["device"] = board.get("device")
            report["board_status"] = status
            report["board"] = board
            report["status"] = "COMPLETED"
            report["result"] = self._summary_result(board)
            self._set_progress(job_id, 100, "COMPLETE", "Đã đọc trạng thái board; không ghi NVS")
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def _load_queries(self) -> list[dict[str, Any]]:
        records = demo.load_prepared(demo.DEFAULT_PREPARED)
        queries = [record for record in records if record.get("role") == "query"]
        if [(item.get("label"), Path(item["source"]["path"]).name) for item in queries] != [
            ("Quang", "Quang_test.m4a"),
            ("Tien", "Tien_test.m4a"),
        ]:
            raise DashboardError("Frozen query set is not exactly Quang/Tien")
        return queries

    @staticmethod
    def _query_result(item: dict[str, Any], raw: bytes, scores: list[dict[str, Any]], reply: dict[str, Any], roundtrip_us: int) -> dict[str, Any]:
        top1 = reply.get("top1")
        expected = item.get("label")
        clean_scores = []
        for score in scores:
            clean_scores.append({
                "label": score.get("label"),
                "cosine": _json_number(score.get("cosine")),
            })
        invoke_us = reply.get("invoke_us")
        if type(invoke_us) is not int or invoke_us < 0:
            invoke_us = None
        return {
            "file": Path(item["source"]["path"]).name,
            "expected_label": expected,
            "top1": top1,
            "correct": type(top1) is str and top1 == expected,
            "scores": clean_scores,
            "encoder_invoke_us": invoke_us if isinstance(invoke_us, int) else None,
            "output_bytes": len(raw),
            "performance": _query_performance(reply, roundtrip_us),
        }

    def _run_frozen(self, job_id: str, output: Path, report: dict[str, Any]) -> None:
        self._set_progress(job_id, 8, "PREPARE", "Đang xác thực manifest và 2 query cố định…")
        queries = self._load_queries()
        report["query_set"] = [Path(item["source"]["path"]).name for item in queries]
        report["query_policy"] = "FROZEN_QUERY_ONLY_NO_ENROLLMENT_WRITE"
        self._set_progress(job_id, 18, "CONNECTING", "Đang kết nối đúng USB Serial/JTAG…")
        client = self._client_factory(output)
        try:
            self._set_progress(job_id, 27, "STATUS_READONLY", "Đọc STATUS read-only trước khi gửi query…")
            board_status = client.status()
            board = _board_summary(client, board_status)
            self._set_board(board)
            report["device"] = board.get("device")
            report["board_status"] = board_status
            report["board"] = board
            _require_ready(board)
            self._set_progress(job_id, 35, "PARITY", "Kiểm tra 5 fixture encoder trên board…")
            parity_results = demo.parity(client)
            report["parity"] = [
                {
                    "fixture_number": item.get("fixture_number"),
                    "max_absdiff_lsb": item.get("max_absdiff_lsb"),
                }
                for item in parity_results
            ]
            results = []
            for index, item in enumerate(queries, start=1):
                self._set_progress(
                    job_id,
                    40 + index * 25,
                    "QUERY",
                    f"Board đang xử lý query {index}/2: {Path(item['source']['path']).name}",
                    query_index=index,
                )
                payload = Path(item["input"]["path"]).read_bytes()
                request_started = time.perf_counter_ns()
                raw, scores, reply = client.recognize(payload)
                roundtrip_us = (time.perf_counter_ns() - request_started) // 1000
                results.append(self._query_result(item, raw, scores, reply, roundtrip_us))
            matched = sum(1 for item in results if item["correct"] is True)
            report["queries"] = results
            report["result"] = {
                "kind": "frozen_queries",
                "query_count": len(results),
                "closed_set_matches": matched,
                "queries": results,
                "accuracy_claim": False,
            }
            report["status"] = "COMPLETED"
            self._set_progress(job_id, 100, "COMPLETE", "Đã nhận 2 kết quả từ board; không phải accuracy claim")
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def _record_with_progress(self, job_id: str, destination: Path) -> Any:
        """Record through a phase-aware recorder without guessing stream start."""

        failure: list[BaseException] = []
        returned: list[Any] = []
        stream_opened = threading.Event()
        capture_started = threading.Event()
        capture_started_at: list[Optional[float]] = [None]
        capture_started_lock = threading.Lock()

        def invoke() -> None:
            try:
                returned.append(self._record_fn(
                    destination,
                    device=self.input_device,
                    phase_callback=phase_callback,
                ))
            except BaseException as error:  # pass the actual mic failure to job state
                failure.append(error)

        def phase_callback(event: str) -> None:
            if event == "opened":
                stream_opened.set()
                self._set_progress(
                    job_id,
                    48,
                    "MIC_PREPARE",
                    "Input PCM Mac đã mở; chờ stream bắt đầu, chưa thu…",
                    recorded_seconds=0.0,
                )
            elif event == "capture_started":
                with capture_started_lock:
                    if capture_started_at[0] is None:
                        capture_started_at[0] = time.monotonic()
                capture_started.set()
                self._set_progress(
                    job_id,
                    50,
                    "MIC_RECORDING",
                    "THU ÂM — nói trong 3 giây (3.0s còn lại)",
                    recorded_seconds=0.0,
                    recording_window_seconds=AUDIO_DURATION_SECONDS,
                )

        self._set_progress(
            job_id,
            47,
            "MIC_PERMISSION",
            "Đang xin quyền microphone Mac; chưa bắt đầu ghi âm…",
            recorded_seconds=0.0,
        )
        thread = threading.Thread(target=invoke, name=f"edgeai-dashboard-mic-{job_id}", daemon=True)
        thread.start()
        while thread.is_alive():
            if not capture_started.is_set():
                if stream_opened.is_set():
                    self._set_progress(
                        job_id,
                        48,
                        "MIC_PREPARE",
                        "Input PCM Mac đã mở; chờ stream bắt đầu, chưa thu…",
                        recorded_seconds=0.0,
                    )
                else:
                    self._set_progress(
                        job_id,
                        47,
                        "MIC_PERMISSION",
                        "Đang xin quyền microphone Mac; chưa bắt đầu ghi âm…",
                        recorded_seconds=0.0,
                    )
            else:
                with capture_started_lock:
                    started = capture_started_at[0]
                elapsed = max(0.0, time.monotonic() - (started or time.monotonic()))
                recorded = min(float(AUDIO_DURATION_SECONDS), elapsed)
                remaining = max(0.0, float(AUDIO_DURATION_SECONDS) - recorded)
                self._set_progress(
                    job_id,
                    50 + int(recorded / AUDIO_DURATION_SECONDS * 15),
                    "MIC_RECORDING",
                    f"THU ÂM — nói trong 3 giây ({remaining:.1f}s còn lại)",
                    recorded_seconds=round(recorded, 1),
                    recording_window_seconds=AUDIO_DURATION_SECONDS,
                )
            thread.join(0.1)
        thread.join()
        if failure:
            raise failure[0]
        if not capture_started.is_set():
            raise DashboardError("Microphone recorder ended before capture started")
        self._set_progress(job_id, 67, "PCM_TRANSFER", "Đã thu PCM thô; không tính đặc trưng trên Mac…")
        return returned[0] if returned else None

    def _recording_metadata(
        self,
        wav_path: Path,
        payload: bytes,
        recorder_metadata: Any,
    ) -> dict[str, Any]:
        """Combine verified WAV/PCM bindings with recorder provenance.

        An injected recorder is useful for host-only tests, but it cannot
        establish native microphone/device provenance.  Its bytes still have
        to satisfy the exact WAV contract, while its source remains explicitly
        unverified in the report.
        """

        binding = _pcm_binding(wav_path, payload)
        native_metadata = recorder_metadata if self._native_recorder and isinstance(recorder_metadata, dict) else None
        injected_metadata = recorder_metadata if not self._native_recorder and isinstance(recorder_metadata, dict) else {}
        if self._native_recorder and native_metadata is None:
            raise DashboardError("Native raw recorder did not return capture metadata")
        if native_metadata is None:
            profile_id = injected_metadata.get("profile_id") if isinstance(injected_metadata.get("profile_id"), str) else "UNVERIFIED_INJECTED_RECORDER"
            provenance = INJECTED_INPUT_PROVENANCE
            provenance_verified = False
            input_device = injected_metadata.get("input_device") if isinstance(injected_metadata.get("input_device"), dict) else None
            requested_format = injected_metadata.get("requested_format") if isinstance(injected_metadata.get("requested_format"), dict) else None
            effective_format = injected_metadata.get("effective_format") if isinstance(injected_metadata.get("effective_format"), dict) else None
            portaudio_requests = injected_metadata.get("portaudio_requests") if isinstance(injected_metadata.get("portaudio_requests"), dict) else None
            callback_count = None
        else:
            profile_id = native_metadata.get("profile_id")
            provenance = native_metadata.get("provenance")
            provenance_verified = native_metadata.get("provenance_verified") is True
            input_device = native_metadata.get("input_device")
            requested_format = native_metadata.get("requested_format")
            effective_format = native_metadata.get("effective_format")
            portaudio_requests = native_metadata.get("portaudio_requests")
            callback_count = native_metadata.get("callback_count") if type(native_metadata.get("callback_count")) is int else None
            if profile_id != RAW_INPUT_PROFILE_ID or provenance != RAW_INPUT_PROVENANCE or not provenance_verified:
                raise DashboardError("Native raw recorder returned invalid provenance metadata")
        actual = {
            "profile_id": profile_id,
            "provenance": provenance,
            "provenance_verified": provenance_verified,
            "input_device": input_device,
            "requested_format": requested_format,
            "effective_format": effective_format,
            "sample_rate_hz": AUDIO_SAMPLE_RATE,
            "channels": AUDIO_CHANNELS,
            "sample_format": "PCM16_LE",
            "format": "PCM16_LE_MONO",
            "sample_count": AUDIO_FRAME_COUNT,
            "pcm_bytes": len(payload),
            "callback_count": callback_count,
            "duration_seconds": AUDIO_DURATION_SECONDS,
            "physical_mic": False,
            "acoustic_equivalence": ACOUSTIC_EQUIVALENCE,
            "acoustic_equivalence_note": "Not calibrated or claimed equivalent to INMP441",
            "os_device_dsp": "UNKNOWN",
            "coreaudio_conversion": "POSSIBLE" if native_metadata is not None else "UNKNOWN",
            "portaudio_requests": portaudio_requests,
            "transport_mode": "BATCH_CAPTURE_THEN_RECOGNIZE_PCM",
            **binding,
        }
        if native_metadata is None:
            actual["provenance_note"] = "Synthetic/injected recorder; not evidence of a native microphone"
        return actual

    def _run_live(self, job_id: str, output: Path, report: dict[str, Any]) -> None:
        self._set_progress(job_id, 8, "CONNECTING", "Đang kết nối board; live không đọc enrollment…")
        client = self._client_factory(output)
        try:
            self._set_progress(job_id, 18, "STATUS_READONLY", "Đọc STATUS read-only; không ghi NVS…")
            board_status = client.status()
            board = _board_summary(client, board_status)
            self._set_board(board)
            report["device"] = board.get("device")
            report["board_status"] = board_status
            report["board"] = board
            _require_ready(board)
            wav16 = output / "live-raw-16k.wav"
            recorder_metadata = self._record_with_progress(job_id, wav16)
            payload = demo.read_pcm_capture(wav16)
            report["recording"] = {
                "file": wav16.name,
                "local_only": True,
                **self._recording_metadata(wav16, payload, recorder_metadata),
            }
            report["processing"] = {
                "mac": "RAW_PCM_CAPTURE_ONLY_NO_APPLICATION_DSP",
                "board": "LOGMEL_CMN_INT8_ENCODER_NVS_COSINE",
                "microphone": "MAC_CAPTURE_SUBSTITUTE_NOT_DIRECT_INMP441",
                "mode": "BATCH_CAPTURE_3S_THEN_SEND_PCM_TO_BOARD",
            }
            self._set_progress(job_id, 75, "BOARD_INFER", "Gửi PCM thô; board log-mel → INT8 → encoder → cosine…")
            request_started = time.perf_counter_ns()
            raw, scores, reply = client.recognize_pcm(payload)
            roundtrip_us = (time.perf_counter_ns() - request_started) // 1000
            clean_scores = [
                {"label": score.get("label"), "cosine": _json_number(score.get("cosine"))}
                for score in scores
            ]
            result = {
                "kind": "live_mac_query",
                "top1": reply.get("top1"),
                "scores": clean_scores,
                "encoder_invoke_us": reply.get("invoke_us") if type(reply.get("invoke_us")) is int and reply.get("invoke_us") >= 0 else None,
                "correct": None,
                "ground_truth": "NOT_PROVIDED",
                "capture": copy.deepcopy(report["recording"]),
                "performance": _query_performance(reply, roundtrip_us),
            }
            report["live"] = {
                "recording": report["recording"],
                "preprocessing": "BOARD_FRONTEND; no application DSP/embedding on Mac",
                "pcm": report["recording"]["wav_binding"],
                "raw_pcm": {
                    "bytes": report["recording"]["pcm_bytes"],
                    "sha256": report["recording"]["pcm_sha256"],
                    "crc32": report["recording"]["pcm_crc32"],
                },
                "response": reply,
                "top1": reply.get("top1"),
                "scores": clean_scores,
                "encoder_invoke_us": result["encoder_invoke_us"],
                "output_bytes": len(raw),
            }
            report["result"] = result
            report["status"] = "COMPLETED"
            self._set_progress(job_id, 100, "COMPLETE", "Board đã trả top-1; chưa có nhãn để tính đúng/sai")
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def _finish_job(self, job_id: str, *, status: str, report_path: Optional[str], error: Optional[str], result: Any) -> None:
        with self._lock:
            if self._active is None or self._active["id"] != job_id:
                return
            job = self._active
            job["status"] = status
            job["finished_at"] = _utc_now()
            job["report_path"] = report_path
            job["error"] = error
            job["result"] = copy.deepcopy(result)
            if status == "COMPLETED":
                job["progress"] = 100
                job["phase"] = "COMPLETE"
            else:
                job["phase"] = "ERROR"
                job["message"] = "Tác vụ lỗi; không dùng dữ liệu giả/fallback"
            summary = copy.deepcopy(job)
            self._last_job = summary
            self._history.insert(0, summary)
            del self._history[MAX_HISTORY:]
            self._active = None

    def _run_job(self, job_id: str, action: str, output: Path) -> None:
        report: dict[str, Any] = {
            "scope": "LOCAL_EDGEAI_DASHBOARD_DEMO_NOT_ACCURACY_BENCHMARK",
            "job_id": job_id,
            "action": action,
            "status": "RUNNING",
            "backend": "usb",
            "output_scope": self._public_path(output),
        }
        report_path: Optional[str] = None
        result: Any = None
        error: Optional[str] = None
        try:
            report = self._base_report(job_id, action, output)
            with self._serial_lock:
                if action == "check_board":
                    self._run_status(job_id, output, report)
                elif action == "run_frozen":
                    self._run_frozen(job_id, output, report)
                elif action == "run_live":
                    self._run_live(job_id, output, report)
                else:
                    raise RequestError("Unknown dashboard action")
            result = report.get("result")
        except BaseException as caught:
            error = _user_error(caught)
            report["status"] = "ERROR"
            report["error"] = error
            self._set_connection("OFFLINE" if _is_offline_error(caught) else "ERROR")
            with self._lock:
                self._board = None
                self._last_error = error
        finally:
            try:
                report_path = self._write_report(output, report)
            except BaseException as report_error:
                report_path = None
                if error is None:
                    error = _user_error(report_error, allow_offline_hint=False)
                report["status"] = "ERROR"
                report["error"] = error
            status = "COMPLETED" if report.get("status") == "COMPLETED" and error is None else "ERROR"
            if status != "COMPLETED":
                result = None
                report.pop("result", None)
            self._finish_job(job_id, status=status, report_path=report_path, error=error, result=result)

    def wait_for_idle(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._active is None:
                    return True
            time.sleep(0.01)
        return False


class DashboardHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server bound by the caller to the loopback address."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], service: DashboardService):
        self.service = service
        super().__init__(server_address, DashboardHandler)


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "EdgeAI-Dashboard"
    sys_version = ""

    @property
    def service(self) -> DashboardService:
        return self.server.service  # type: ignore[attr-defined]

    @property
    def server_port(self) -> int:
        return int(self.server.server_port)

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the terminal useful without echoing request bodies or paths.
        print(f"[dashboard] {self.address_string()} {format % args}", flush=True)

    def _allowed_hosts(self) -> set[str]:
        return {
            f"{HOST}:{self.server_port}",
            f"localhost:{self.server_port}",
        }

    def _allowed_origins(self) -> set[str]:
        return {
            f"http://{HOST}:{self.server_port}",
            f"http://localhost:{self.server_port}",
        }

    def _validate_request_context(self, *, mutation: bool) -> None:
        host = (self.headers.get("Host") or "").strip().lower()
        if host not in self._allowed_hosts():
            raise RequestError("Host không thuộc loopback dashboard")
        origin = self.headers.get("Origin")
        if origin and origin not in self._allowed_origins():
            raise RequestError("Origin không hợp lệ")
        if mutation and origin not in self._allowed_origins():
            raise RequestError("POST cần Origin loopback hợp lệ")

    def _security_headers(self, content_type: str) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")

    def _send_json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self._security_headers("application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"error": _redact_text(message), "status": "ERROR"})

    def _path(self) -> str:
        parsed = urlsplit(self.path)
        if parsed.query or parsed.fragment:
            raise RequestError("Query parameters are not accepted")
        if not parsed.path.startswith("/") or "\\" in parsed.path:
            raise RequestError("Invalid request path")
        return parsed.path

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            self._validate_request_context(mutation=False)
            path = self._path()
            if path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self._security_headers("image/x-icon")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if path == "/api/session":
                self._send_json(HTTPStatus.OK, self.service.session())
                return
            if path == "/api/state":
                self._send_json(HTTPStatus.OK, self.service.state())
                return
            if path not in STATIC_FILES:
                self._send_error_json(HTTPStatus.NOT_FOUND, "Not found")
                return
            filename, content_type = STATIC_FILES[path]
            file_path = (WEB_ROOT / filename).resolve()
            if file_path.parent != WEB_ROOT.resolve() or not file_path.is_file():
                self._send_error_json(HTTPStatus.NOT_FOUND, "Static asset unavailable")
                return
            body = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self._security_headers(content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except RequestError as error:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as error:  # keep server alive on malformed local requests
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

    def _read_json_body(self) -> dict[str, Any]:
        length_text = self.headers.get("Content-Length")
        if length_text is None:
            raise RequestError("Content-Length is required")
        try:
            length = int(length_text)
        except ValueError as error:
            raise RequestError("Invalid Content-Length") from error
        if length < 0:
            raise RequestError("Invalid Content-Length")
        if length > MAX_JSON_BODY:
            raise PayloadLimitError("JSON body exceeds limit")
        transfer = (self.headers.get("Transfer-Encoding") or "").lower()
        if transfer and transfer != "identity":
            raise RequestError("Chunked transfer is not accepted")
        content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise RequestError("Content-Type must be application/json")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise RequestError("Incomplete JSON body")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RequestError("Malformed JSON body") from error
        if not isinstance(value, dict) or set(value) != {"action"} or not isinstance(value["action"], str):
            raise RequestError("JSON body must contain only action")
        return value

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            self._validate_request_context(mutation=True)
            path = self._path()
            if path != "/api/action":
                self._send_error_json(HTTPStatus.NOT_FOUND, "Not found")
                return
            token = self.headers.get("X-CSRF-Token") or ""
            if not secrets.compare_digest(token, self.service.csrf_token):
                self._send_error_json(HTTPStatus.FORBIDDEN, "CSRF token không hợp lệ")
                return
            body = self._read_json_body()
            try:
                job = self.service.start_action(body["action"])
            except BusyError as error:
                self._send_error_json(HTTPStatus.CONFLICT, str(error))
                return
            except RequestError as error:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
                return
            self._send_json(HTTPStatus.ACCEPTED, {"status": "ACCEPTED", "job": job})
        except RequestError as error:
            message = str(error)
            if isinstance(error, PayloadLimitError):
                status = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
            else:
                status = HTTPStatus.FORBIDDEN if "Origin" in message or "CSRF" in message else HTTPStatus.BAD_REQUEST
            self._send_error_json(status, message)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as error:
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))


def make_server(service: Optional[DashboardService] = None, *, port: int = PORT) -> DashboardHTTPServer:
    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
        raise ValueError("Port must be between 0 and 65535")
    app = service or DashboardService()
    server = DashboardHTTPServer((HOST, port), app)
    app.port = int(server.server_port)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=PORT, help="Loopback port; default 8765")
    parser.add_argument(
        "--input-device",
        default=None,
        help="Local sounddevice input name or numeric index; default is the system input",
    )
    args = parser.parse_args()
    server = make_server(port=args.port, service=DashboardService(input_device=args.input_device))
    print(f"EdgeAI dashboard: http://{HOST}:{server.server_port}/", flush=True)
    print("Hardware and microphone start only after an explicit button action.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping dashboard", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
