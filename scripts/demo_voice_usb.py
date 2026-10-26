#!/usr/bin/env python3
"""Local voice demo: raw PCM live path, legacy tensor diagnostics over USB.

Original recordings and frozen training artifacts are never changed. The
explicit host backend and optional Mac score check are diagnostics, never a
USB fallback or the live board decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import secrets
import struct
import subprocess
import sys
import time
import wave
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from edgeai.frontend import load_wav, log_mel
from edgeai.matching import cosine_similarity, enroll, normalize_embedding

MODEL_SHA = "aa42a3140f375f3a2c93648893132b880dd3e9f4be4ca8f82cde09fc6c440727"
MANIFEST_SHA = "4af129472ed9740ade5e66c1a68800f389e6b4c7f7d4de2f7b80dd59a0102a4f"
VOICE_MANIFEST_SHA = "1d73cad6e9697999f1cfa1ed0a672b12fd8fd0c533c34dab7b6afe622561baa7"
GENERATED = ROOT / "firmware/esp32_s3_smoke/generated"
MODEL = GENERATED / "encoder_int8.tflite"
# A source checkout already includes this exact hash-pinned model. Generated
# smoke inputs remain a local build prerequisite, not a live-query prerequisite.
if not MODEL.is_file():
    MODEL = ROOT / "models/full100-20261015/baseline_kd/encoder_int8.tflite"
MAGIC = b"EDGEAI01"
HEADER = struct.Struct("<8sIII")
PREFIX = b"EDGEAI_RESPONSE "
OPERATION_VERSION = 1
OP_ENCODE = 0x01
OP_PROVISION = 0x10
OP_STATUS = 0x11
OP_RECOGNIZE = 0x12
OP_ADD_AN = 0x13
OP_RECOGNIZE_PCM = 0x20
OP_APPEND_PCM = 0x21
PCM_SAMPLE_RATE = 16000
PCM_SAMPLES = 48000
PCM_BYTES = PCM_SAMPLES * 2
PCM_PAYLOAD_BYTES = 2 + PCM_BYTES
LABEL_BYTES = 8
MAX_TEMPLATES = 5
APPEND_PCM_PAYLOAD_BYTES = 2 + LABEL_BYTES + PCM_BYTES
FRONTEND_ID = "LOGMEL_16K_400_160_512_40_CMN_V1"
INPUT_BYTES = 11920
OUTPUT_BYTES = 192
ENCODE_PAYLOAD_BYTES = 2 + INPUT_BYTES
RECOGNIZE_PAYLOAD_BYTES = 2 + INPUT_BYTES
ADD_AN_PAYLOAD_BYTES = 2 + INPUT_BYTES
PROVISION_PAYLOAD_BYTES = 2 + 2 * INPUT_BYTES
STATUS_PAYLOAD_BYTES = 2
MAX_PAYLOAD_BYTES = APPEND_PCM_PAYLOAD_BYTES
INPUT_SCALE = 0.12695927917957306
INPUT_ZERO = -12
OUTPUT_SCALE = 0.01833495683968067
OUTPUT_ZERO = 6
SERIAL = "7C:0C:5F:5B:0E:00"
DEFAULT_PREPARED = ROOT / "artifacts/voice-demo/20261019/prepared"
DEFAULT_AN_PREPARED = ROOT / "artifacts/voice-demo/20261020/an-prepared"
AN_SOURCE_SHA = "43d16865db83a96e154449530fc81eba53b8e2ace2ff640e5c0237aeff64fe8b"
AN_PREPROCESSING = "afconvert mono PCM16 16kHz; first48000/pad_right; locked frontend"
FILES = (("Quang", "enroll", "Quang.m4a"), ("Tien", "enroll", "Tien.m4a"),
         ("Quang", "query", "Quang_test.m4a"), ("Tien", "query", "Tien_test.m4a"))
AN_FILE = ("An", "enroll", "An.m4a")


def binding(path: Path) -> dict:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def new_output(path: Path) -> Path:
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    return path


def convert_audio(source: Path, destination: Path) -> None:
    if destination.exists():
        raise ValueError("Refusing audio overwrite")
    converted = destination.with_suffix(".afconvert.wav")
    if converted.exists():
        raise ValueError("Refusing converter intermediate overwrite")
    subprocess.run(["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@16000",
                    "-c", "1", str(source), str(converted)], check=True,
                   capture_output=True, timeout=30)
    # CoreAudio writes WAVE_FORMAT_EXTENSIBLE; Python 3.11 wave only reads
    # canonical PCM. Validate the PCM subtype then rewrite its header, not audio.
    data = converted.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Converter did not produce RIFF/WAVE")
    fmt, pcm = None, None
    offset = 12
    while offset + 8 <= len(data):
        kind, size = struct.unpack_from("<4sI", data, offset)
        start = offset + 8
        if start + size > len(data):
            raise ValueError("Truncated converter WAV")
        if kind == b"fmt ":
            fmt = data[start:start + size]
        elif kind == b"data":
            pcm = data[start:start + size]
        offset = start + size + (size % 2)
    if fmt is None or len(fmt) < 16 or pcm is None or len(pcm) % 2:
        raise ValueError("Invalid converter WAV chunks")
    tag, channels, rate, byte_rate, align, bits = struct.unpack_from("<HHIIHH", fmt)
    if (channels, rate, byte_rate, align, bits) != (1, 16000, 32000, 2, 16):
        raise ValueError("Unexpected converter audio format")
    if tag == 65534:
        pcm_guid = bytes.fromhex("0100000000001000800000aa00389b71")
        if len(fmt) < 40 or struct.unpack_from("<H", fmt, 16)[0] < 22 or fmt[24:40] != pcm_guid:
            raise ValueError("Non-PCM extensible WAV")
    elif tag != 1:
        raise ValueError("Non-PCM WAV")
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(pcm)


def quantize(features: np.ndarray) -> bytes:
    values = np.asarray(features, dtype=np.float32)
    if values.shape != (298, 40) or not np.isfinite(values).all():
        raise ValueError("Expected finite [298,40] features")
    # Match frozen feature preparation: float32 division, ties-to-even rint.
    return np.clip(np.rint(values / INPUT_SCALE) + INPUT_ZERO,
                   -128, 127).astype(np.int8).tobytes()


def read_pcm_capture(path: Path) -> bytes:
    """Strip a WAV container only; never resample, normalize, crop or pad.

    A 16 kHz mono PCM16 acquisition is the transport contract. The OS/audio
    device may implement the requested acquisition rate; this is not a claim
    that a microphone is attached to the MCU.
    """
    with wave.open(str(path), "rb") as handle:
        if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth(),
                handle.getnframes(), handle.getcomptype()) != (
                PCM_SAMPLE_RATE, 1, 2, PCM_SAMPLES, "NONE"):
            raise ValueError("Raw capture requires exactly 3s mono PCM16 16kHz; no host conversion")
        payload = handle.readframes(PCM_SAMPLES)
    if len(payload) != PCM_BYTES:
        raise ValueError("Truncated raw PCM capture")
    return payload


def prepare(data: Path, output: Path) -> None:
    out = new_output(output)
    records = []
    for label, role, filename in FILES:
        source = data / filename
        wav = out / (source.stem + ".wav")
        before = binding(source)
        convert_audio(source, wav)
        pcm, report = load_wav(wav)
        payload = quantize(log_mel(pcm))
        input_file = out / (source.stem + ".int8")
        with input_file.open("xb") as handle:
            handle.write(payload)
        records.append({"label": label, "role": role, "source": before,
                        "wav": binding(wav), "input": binding(input_file),
                        "wav_load": report.as_dict(),
                        "pcm_rms": float(np.sqrt(np.mean(pcm.astype(np.float64)**2))),
                        "pcm_peak": float(np.max(np.abs(pcm))),
                        "pcm_clipped_fraction": float(np.mean(np.abs(pcm) >= .999)),
                        "input_crc32": zlib.crc32(payload)})
        if binding(source) != before:
            raise ValueError("Source changed during preparation")
    write_json(out / "manifest.json", {
        "scope": "PERSONAL_TWO_SPEAKER_DEMO_NOT_BENCHMARK", "records": records,
        "preprocessing": AN_PREPROCESSING,
        "model_sha256": MODEL_SHA, "query_used_for_enrollment": False})
    print("Prepared:", out)


def prepare_an(data: Path, output: Path) -> None:
    """Prepare the separately pinned An enrollment input, never a query."""
    source = data / AN_FILE[2]
    before = binding(source)
    if before["sha256"] != AN_SOURCE_SHA:
        raise ValueError("An source hash mismatch")
    out = new_output(output)
    wav = out / "An.wav"
    convert_audio(source, wav)
    pcm, report = load_wav(wav)
    payload = quantize(log_mel(pcm))
    input_file = out / "An.int8"
    with input_file.open("xb") as handle:
        handle.write(payload)
    if binding(source) != before:
        raise ValueError("An source changed during preparation")
    record = {"label": "An", "role": "enroll", "source": before,
              "wav": binding(wav), "input": binding(input_file),
              "wav_load": report.as_dict(),
              "pcm_rms": float(np.sqrt(np.mean(pcm.astype(np.float64)**2))),
              "pcm_peak": float(np.max(np.abs(pcm))),
              "pcm_clipped_fraction": float(np.mean(np.abs(pcm) >= .999)),
              "input_crc32": zlib.crc32(payload)}
    write_json(out / "manifest.json", {
        "scope": "EXPLICIT_AN_ENROLLMENT_NOT_QUERY_OR_BENCHMARK",
        "model_sha256": MODEL_SHA, "source_sha256": AN_SOURCE_SHA,
        "query_used_for_enrollment": False,
        "preprocessing": "afconvert mono PCM16 16kHz; first48000/pad_right; locked frontend",
        "records": [record]})
    print("Prepared An:", out)


def operation_payload(opcode: int, body: bytes = b"") -> bytes:
    payload = bytes((opcode, OPERATION_VERSION)) + body
    expected = {
        OP_ENCODE: ENCODE_PAYLOAD_BYTES,
        OP_PROVISION: PROVISION_PAYLOAD_BYTES,
        OP_STATUS: STATUS_PAYLOAD_BYTES,
        OP_RECOGNIZE: RECOGNIZE_PAYLOAD_BYTES,
        OP_ADD_AN: ADD_AN_PAYLOAD_BYTES,
        OP_RECOGNIZE_PCM: PCM_PAYLOAD_BYTES,
        OP_APPEND_PCM: APPEND_PCM_PAYLOAD_BYTES,
    }.get(opcode)
    if expected is None or len(payload) != expected:
        raise ValueError("Invalid operation payload size")
    return payload


def frame(request_id: int, payload: bytes) -> bytes:
    if len(payload) not in (0, INPUT_BYTES, ENCODE_PAYLOAD_BYTES,
                            RECOGNIZE_PAYLOAD_BYTES, ADD_AN_PAYLOAD_BYTES, PROVISION_PAYLOAD_BYTES,
                            STATUS_PAYLOAD_BYTES, PCM_PAYLOAD_BYTES, APPEND_PCM_PAYLOAD_BYTES):
        raise ValueError("Invalid request size")
    return HEADER.pack(MAGIC, request_id, len(payload), zlib.crc32(payload)) + payload


_TELEMETRY_BASE_FIELDS = (
    "version", "uptime_us", "internal_heap_total_bytes",
    "internal_heap_free_bytes", "internal_heap_min_free_bytes",
    "internal_heap_largest_free_block_bytes",
)
_TELEMETRY_QUERY_FIELDS = (
    "query_heap_free_before_bytes", "query_heap_free_after_bytes",
    "query_heap_delta_bytes", "recognition_us",
)


def validate_telemetry(reply: dict) -> dict[str, int] | None:
    """Validate and normalize optional board telemetry.

    A missing telemetry object is the compatibility path for older firmware.
    Present telemetry is deliberately strict so a dashboard never displays a
    partially trusted memory or timing sample.
    """
    if not isinstance(reply, dict):
        raise ValueError("Malformed board reply")
    if "telemetry" not in reply:
        return None
    telemetry = reply["telemetry"]
    if not isinstance(telemetry, dict):
        raise ValueError("Malformed telemetry object")

    query_present = any(key in telemetry for key in _TELEMETRY_QUERY_FIELDS)
    required = set(_TELEMETRY_BASE_FIELDS)
    if query_present:
        required.update(_TELEMETRY_QUERY_FIELDS)
    actual = set(telemetry)
    missing = required - actual
    unknown = actual - required
    if missing or unknown:
        raise ValueError("Malformed telemetry fields")
    if type(telemetry["version"]) is not int or telemetry["version"] != 1:
        raise ValueError("Unknown telemetry version")

    nonnegative_fields = _TELEMETRY_BASE_FIELDS
    if query_present:
        nonnegative_fields += (
            "query_heap_free_before_bytes", "query_heap_free_after_bytes",
            "recognition_us",
        )
    for key in nonnegative_fields:
        value = telemetry[key]
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid telemetry integer: {key}")
    if query_present and type(telemetry["query_heap_delta_bytes"]) is not int:
        raise ValueError("Invalid telemetry integer: query_heap_delta_bytes")

    total = telemetry["internal_heap_total_bytes"]
    free = telemetry["internal_heap_free_bytes"]
    minimum = telemetry["internal_heap_min_free_bytes"]
    largest = telemetry["internal_heap_largest_free_block_bytes"]
    if free > total or minimum > free or largest > free:
        raise ValueError("Impossible telemetry heap relationship")

    normalized = {key: telemetry[key] for key in _TELEMETRY_BASE_FIELDS}
    if query_present:
        before = telemetry["query_heap_free_before_bytes"]
        after = telemetry["query_heap_free_after_bytes"]
        if before > total or after > total:
            raise ValueError("Query heap sample exceeds telemetry capacity")
        if after != free:
            raise ValueError("Query heap after differs from telemetry snapshot")
        if telemetry["query_heap_delta_bytes"] != before - after:
            raise ValueError("Query heap delta is inconsistent")
        recognition_us = telemetry["recognition_us"]
        if recognition_us <= 0:
            raise ValueError("Invalid recognition timing")
        invoke_us = reply.get("invoke_us")
        if reply.get("operation") == "RECOGNIZE" or "invoke_us" in reply:
            if type(invoke_us) is not int or invoke_us <= 0:
                raise ValueError("Invalid invoke timing")
            if recognition_us < invoke_us:
                raise ValueError("Recognition timing excludes invoke timing")
        normalized.update({key: telemetry[key] for key in _TELEMETRY_QUERY_FIELDS})
    return normalized


def check_metadata(reply: dict) -> None:
    expected = {"protocol_version": 1, "model_sha256": MODEL_SHA,
                "runtime_mode": "ESP_NN_ANSI_C_REFERENCE", "input_bytes": INPUT_BYTES,
                "output_bytes": OUTPUT_BYTES, "input_zero_point": INPUT_ZERO,
                "output_zero_point": OUTPUT_ZERO, "arena_capacity_bytes": 131072}
    for key, value in expected.items():
        if type(reply.get(key)) is not type(value) or reply[key] != value:
            raise ValueError(f"Board metadata mismatch: {key}")
    for key, value in (("input_scale", INPUT_SCALE), ("output_scale", OUTPUT_SCALE)):
        if not math.isclose(float(reply.get(key, float("nan"))), value,
                            rel_tol=1e-7, abs_tol=1e-10):
            raise ValueError(f"Board quantization mismatch: {key}")
    used = reply.get("arena_used_bytes")
    if type(used) is not int or not 0 < used <= 131072:
        raise ValueError("Invalid arena evidence")
    validate_telemetry(reply)


def check_operation(reply: dict, expected: str) -> None:
    if reply.get("operation") != expected:
        raise ValueError(f"Board operation mismatch: {reply.get('operation')}")


def check_persistence_status(reply: dict) -> None:
    if reply.get("persistence_source") != "NVS_BOOT_READONLY":
        raise ValueError("Board status did not identify read-only boot persistence")
    state = reply.get("persistence_state")
    if state not in {"EMPTY", "READY", "UNAVAILABLE", "CORRUPT",
                     "PERSISTENCE_UNCERTAIN", "NOT_INITIALIZED"}:
        raise ValueError("Unknown board persistence state")
    record_crc32 = reply.get("record_crc32")
    if type(record_crc32) is not int or not 0 <= record_crc32 <= 0xffffffff:
        raise ValueError("Invalid persisted record fingerprint CRC")
    fingerprint = reply.get("record_fingerprint")
    expected = f"crc32:{record_crc32:08x}"
    if fingerprint != expected:
        raise ValueError("Persisted record fingerprint is not deterministic")
    count = reply.get("enrollment_count")
    labels = reply.get("labels")
    if reply.get("persistence_state") == "READY":
        if not isinstance(count, int) or not 2 <= count <= MAX_TEMPLATES or not _valid_active_labels(labels, count):
            raise ValueError("Board active-label set is invalid")
    elif labels != []:
        raise ValueError("Inactive board status must not expose labels")


def response_output(reply: dict, request_id: int, payload: bytes,
                    operation: str = "ENCODE") -> bytes:
    check_metadata(reply)
    if type(reply.get("request_id")) is not int or reply["request_id"] != request_id:
        raise ValueError("Stale/mismatched response ID")
    check_operation(reply, operation)
    if reply.get("status") != "OK" or reply.get("input_crc32") != zlib.crc32(payload):
        raise ValueError("Board rejected input or CRC mismatch")
    text = reply.get("output_hex", "")
    if not isinstance(text, str) or len(text) != 384:
        raise ValueError("Expected exactly192 output bytes")
    raw = bytes.fromhex(text)
    if len(raw) != OUTPUT_BYTES:
        raise ValueError("Invalid output bytes")
    if type(reply.get("invoke_us")) is not int or reply["invoke_us"] <= 0:
        raise ValueError("Invalid invoke timing")
    return raw


def recognition_output(reply: dict, request_id: int, payload: bytes,
                       operation: str = "RECOGNIZE", *,
                       expected_labels: list[str] | None = None) -> tuple[bytes, list[dict]]:
    raw = response_output(reply, request_id, payload, operation)
    if operation == "RECOGNIZE_PCM":
        if len(payload) != PCM_BYTES:
            raise ValueError("Invalid PCM response binding")
        for key, value in (("pcm_sample_rate_hz", PCM_SAMPLE_RATE),
                           ("pcm_samples", PCM_SAMPLES), ("frontend_id", FRONTEND_ID)):
            if type(reply.get(key)) is not type(value) or reply[key] != value:
                raise ValueError(f"Board frontend mismatch: {key}")
        if type(reply.get("frontend_us")) is not int or reply["frontend_us"] <= 0:
            raise ValueError("Missing board frontend timing")
        crc = reply.get("feature_crc32")
        if type(crc) is not int or not 0 <= crc <= 0xffffffff:
            raise ValueError("Invalid board feature CRC")
        telemetry = validate_telemetry(reply)
        if telemetry is None or telemetry.get("recognition_us", 0) < reply["frontend_us"] + reply["invoke_us"]:
            raise ValueError("PCM recognition timing must include frontend and encoder")
    scores = reply.get("scores")
    if not isinstance(scores, list) or not 2 <= len(scores) <= MAX_TEMPLATES:
        raise ValueError("Board recognition did not return a valid score set")
    if any(not isinstance(item, dict) for item in scores):
        raise ValueError("Malformed board recognition score")
    labels = [item.get("label") for item in scores]
    # Scores are ranked by cosine, not by NVS registration order.
    required = {"Quang", "Tien"} | ({"An"} if len(scores) >= 3 else set())
    try:
        packed = [safe_label(label) for label in labels]
    except ValueError:
        raise ValueError("Board recognition labels are not the active set") from None
    if len(set(packed)) != len(scores) or not required.issubset(labels):
        raise ValueError("Board recognition labels are not the active set")
    if expected_labels is not None and set(labels) != set(expected_labels):
        raise ValueError("Board recognition labels differ from checked STATUS")
    for item in scores:
        if set(item) != {"label", "cosine"} or not isinstance(item["label"], str):
            raise ValueError("Malformed board recognition score")
        if type(item["cosine"]) not in (int, float) or not math.isfinite(item["cosine"]) or not -1 <= item["cosine"] <= 1:
            raise ValueError("Non-finite board recognition score")
    if any(left["cosine"] < right["cosine"] for left, right in zip(scores, scores[1:])):
        raise ValueError("Board recognition scores are not ranked")
    top1 = reply.get("top1")
    if top1 != scores[0]["label"]:
        raise ValueError("Board top1 is not the first stable-ranked score")
    return raw, scores


def safe_label(label: str) -> bytes:
    if not isinstance(label, str) or not 1 <= len(label) <= LABEL_BYTES - 1:
        raise ValueError("Label must contain 1..7 safe ASCII characters")
    if not label.isascii() or any(not (char.isalnum() or char in "-_") for char in label):
        raise ValueError("Label must use only ASCII letters, digits, '-' or '_'")
    return label.encode("ascii") + b"\0" * (LABEL_BYTES - len(label))


def _valid_active_labels(labels: object, count: int) -> bool:
    if not isinstance(labels, list) or len(labels) != count or labels[:2] != ["Quang", "Tien"]:
        return False
    if count >= 3 and labels[2] != "An":
        return False
    try:
        packed = [safe_label(label) for label in labels]
    except ValueError:
        return False
    return len(set(packed)) == len(packed)


class USBEncoder:
    def __init__(self, serial_number: str, output: Path):
        import serial
        from serial.tools import list_ports
        wanted = serial_number.replace(":", "").lower()
        if not wanted:
            raise ValueError("Empty USB identity")
        ports = [p for p in list_ports.comports()
                 if p.vid == 0x303A and p.pid == 0x1001
                 and (p.serial_number or "").replace(":", "").lower() == wanted]
        if len(ports) != 1:
            raise ValueError("Need exactly one matching ESP32 USB device")
        self.device = {"port": ports[0].device, "serial": ports[0].serial_number,
                       "vid": ports[0].vid, "pid": ports[0].pid}
        self.counter = secrets.randbelow(2**31)
        self.log = (output / "usb-serial.log").open("xb")
        self.port = serial.Serial(port=None, baudrate=115200, timeout=.2, write_timeout=10)
        self.port.dtr = False
        self.port.rts = False
        self.port.port = ports[0].device
        try:
            self.port.open()
            time.sleep(1)
            self.hello = self.request(b"")
            check_metadata(self.hello)
            check_operation(self.hello, "HELLO")
            if self.hello.get("status") != "READY":
                raise ValueError("Board USB service not ready")
        except BaseException:
            self.close()
            raise

    def close(self):
        self.port.close()
        self.log.close()

    def request(self, payload: bytes) -> dict:
        self.counter += 1
        request_id = self.counter
        packet = frame(request_id, payload)
        for start in range(0, len(packet), 256):
            part = packet[start:start+256]
            if self.port.write(part) != len(part):
                raise ValueError("Short USB write")
        self.port.flush()
        deadline = time.monotonic() + 15
        buffered = bytearray()
        while time.monotonic() < deadline:
            chunk = self.port.read(2048)
            if chunk:
                self.log.write(chunk)
                self.log.flush()
                buffered.extend(chunk)
            while b"\n" in buffered:
                line, _, rest = buffered.partition(b"\n")
                buffered = bytearray(rest)
                if not line.startswith(PREFIX):
                    continue
                reply = json.loads(line[len(PREFIX):])
                if reply.get("request_id") != request_id:
                    continue
                check_metadata(reply)
                if reply.get("status") == "ERROR":
                    raise ValueError("Board error: " + str(reply.get("error")))
                return reply
            if len(buffered) > 65536:
                raise ValueError("Oversize serial response")
        raise TimeoutError("No matching USB reply; firmware may need update/reset")

    def infer(self, payload: bytes) -> tuple[bytes, dict]:
        if len(payload) != INPUT_BYTES:
            raise ValueError("Expected exactly one encoder input tensor")
        reply = self.request(operation_payload(OP_ENCODE, payload))
        return response_output(reply, self.counter, payload), reply

    def status(self) -> dict:
        reply = self.request(operation_payload(OP_STATUS))
        check_operation(reply, "STATUS")
        if reply.get("status") != "OK":
            raise ValueError("Board status failed: " + str(reply.get("error")))
        check_persistence_status(reply)
        self.active_labels = reply.get("labels") if reply.get("persistence_state") == "READY" else None
        return reply

    def provision(self, enrollment: list[dict]) -> dict:
        expected = [(label, role) for label, role, _ in FILES[:2]]
        if [(item["label"], item["role"]) for item in enrollment] != expected:
            raise ValueError("Provision requires exactly Quang/Tien enrollment records")
        tensors = []
        for item in enrollment:
            payload = Path(item["input"]["path"]).read_bytes()
            if len(payload) != INPUT_BYTES:
                raise ValueError("Invalid enrollment input tensor size")
            tensors.append(payload)
        reply = self.request(operation_payload(OP_PROVISION, b"".join(tensors)))
        check_operation(reply, "PROVISION")
        if reply.get("status") != "OK":
            raise ValueError("Board provision failed: " + str(reply.get("error")))
        check_persistence_status(reply)
        if reply.get("persistence_state") != "READY" or reply.get("enrollment_count") != 2:
            raise ValueError("Board did not report a ready two-template record")
        return reply

    def add_an(self, enrollment: dict) -> dict:
        if (enrollment.get("label"), enrollment.get("role")) != ("An", "enroll"):
            raise ValueError("ADD_AN requires exactly the independently prepared An enrollment")
        payload = Path(enrollment["input"]["path"]).read_bytes()
        if len(payload) != INPUT_BYTES:
            raise ValueError("Invalid An enrollment input tensor size")
        reply = self.request(operation_payload(OP_ADD_AN, payload))
        check_operation(reply, "ADD_AN")
        if reply.get("status") != "OK":
            raise ValueError("Board ADD_AN failed: " + str(reply.get("error")))
        check_persistence_status(reply)
        if (reply.get("persistence_state"), reply.get("enrollment_count"),
                reply.get("labels")) != ("READY", 3, ["Quang", "Tien", "An"]):
            raise ValueError("Board did not report the three-speaker record")
        return reply

    def recognize(self, payload: bytes) -> tuple[bytes, list[dict], dict]:
        if len(payload) != INPUT_BYTES:
            raise ValueError("Expected exactly one recognition input tensor")
        reply = self.request(operation_payload(OP_RECOGNIZE, payload))
        raw, scores = recognition_output(reply, self.counter, payload,
                                         expected_labels=getattr(self, "active_labels", None))
        return raw, scores, reply

    def recognize_pcm(self, payload: bytes) -> tuple[bytes, list[dict], dict]:
        if len(payload) != PCM_BYTES:
            raise ValueError("Expected exactly 96000 raw PCM16 bytes")
        reply = self.request(operation_payload(OP_RECOGNIZE_PCM, payload))
        raw, scores = recognition_output(reply, self.counter, payload, "RECOGNIZE_PCM",
                                         expected_labels=getattr(self, "active_labels", None))
        return raw, scores, reply

    def append_pcm(self, label: str, payload: bytes) -> dict:
        if len(payload) != PCM_BYTES:
            raise ValueError("APPEND_PCM requires exactly 96000 raw PCM16 bytes")
        append = self.hello.get("append_pcm")
        if not isinstance(append, dict) or append != {
            "operation": OP_APPEND_PCM, "operation_version": OPERATION_VERSION,
            "label_bytes": LABEL_BYTES, "sample_rate_hz": PCM_SAMPLE_RATE,
            "samples": PCM_SAMPLES, "bytes": PCM_BYTES, "max_templates": MAX_TEMPLATES}:
            raise ValueError("Board does not advertise APPEND_PCM capability")
        reply = self.request(operation_payload(OP_APPEND_PCM, safe_label(label) + payload))
        check_operation(reply, "APPEND_PCM")
        if reply.get("status") != "OK":
            raise ValueError("Board APPEND_PCM failed: " + str(reply.get("error")))
        check_persistence_status(reply)
        return reply


class HostEncoder:
    """Explicit offline oracle; never identified as a board run."""
    def __init__(self, output: Path):
        if binding(MODEL)["sha256"] != MODEL_SHA:
            raise ValueError("Model hash mismatch")
        self.out, self.counter = output, 0
        self.device = {"backend": "HOST_TFLM_NOT_BOARD"}

    def infer(self, payload: bytes) -> tuple[bytes, dict]:
        self.counter += 1
        source = self.out / f"host-{self.counter:03d}.input.int8"
        dest = self.out / f"host-{self.counter:03d}.output.int8"
        with source.open("xb") as handle:
            handle.write(payload)
        run = subprocess.run([str(ROOT / "artifacts/tflm-host/edgeai_tflm_host"),
                              str(MODEL), str(source), str(dest), "131072"],
                             capture_output=True, text=True, timeout=30, check=True)
        report = json.loads(run.stdout)
        if report.get("status") != "PASS" or dest.stat().st_size != 192:
            raise ValueError("Host oracle failure")
        return dest.read_bytes(), report

    def close(self):
        pass


def parity(encoder) -> list[dict]:
    if binding(GENERATED / "manifest.json")["sha256"] != MANIFEST_SHA:
        raise ValueError("Frozen fixture manifest changed")
    manifest = json.loads((GENERATED / "manifest.json").read_text())
    results = []
    for item in manifest["fixtures"]:
        src, exp = GENERATED / item["input_file"], GENERATED / item["expected_file"]
        if binding(src)["sha256"] != item["input"]["sha256"] or binding(exp)["sha256"] != item["expected"]["sha256"]:
            raise ValueError("Frozen fixture binding changed")
        raw, reply = encoder.infer(src.read_bytes())
        diff = np.frombuffer(raw, np.int8).astype(np.int16) - np.frombuffer(exp.read_bytes(), np.int8).astype(np.int16)
        maximum = int(np.abs(diff).max())
        results.append({"fixture_number": item["fixture_number"], "max_absdiff_lsb": maximum,
                        "raw_hex": raw.hex(), "response": reply})
        if maximum > 1:
            raise ValueError("USB/host fixture parity failed")
    return results


def embedding(raw: bytes) -> np.ndarray:
    if len(raw) != 192:
        raise ValueError("Invalid embedding size")
    values = (np.frombuffer(raw, np.int8).astype(np.float32) - OUTPUT_ZERO) * np.float32(OUTPUT_SCALE)
    return normalize_embedding(values)


def rank(query: np.ndarray, templates: dict[str, np.ndarray]) -> list[dict]:
    return sorted([{"label": label, "cosine": cosine_similarity(query, template)}
                   for label, template in templates.items()], key=lambda v: (-v["cosine"], v["label"]))


def load_prepared(root: Path) -> list[dict]:
    manifest_bytes = (root / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != VOICE_MANIFEST_SHA:
        raise ValueError("Prepared voice manifest hash mismatch; enrollment set is frozen")
    manifest = json.loads(manifest_bytes)
    if manifest.get("model_sha256") != MODEL_SHA or manifest.get("query_used_for_enrollment") is not False:
        raise ValueError("Invalid prepared manifest")
    records = manifest["records"]
    if [(r["label"], r["role"], Path(r["source"]["path"]).name) for r in records] != list(FILES):
        raise ValueError("Enrollment/query split differs from declared four files")
    for record in records:
        for key in ("source", "wav", "input"):
            if binding(Path(record[key]["path"])) != record[key]:
                raise ValueError(f"Prepared/source file binding changed: {key}")
        if record["input"]["bytes"] != INPUT_BYTES:
            raise ValueError("Wrong prepared input size")
    return records


def load_prepared_an(root: Path) -> dict:
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("scope") != "EXPLICIT_AN_ENROLLMENT_NOT_QUERY_OR_BENCHMARK" or
            manifest.get("model_sha256") != MODEL_SHA or
            manifest.get("source_sha256") != AN_SOURCE_SHA or
            manifest.get("preprocessing") != AN_PREPROCESSING or
            manifest.get("query_used_for_enrollment") is not False):
        raise ValueError("Invalid independently prepared An manifest")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("An manifest must contain exactly one enrollment record")
    record = records[0]
    if (record.get("label"), record.get("role"),
            Path(record.get("source", {}).get("path", "")).name) != AN_FILE:
        raise ValueError("An enrollment record differs from the declared source")
    for key in ("source", "wav", "input"):
        if not isinstance(record.get(key), dict) or binding(Path(record[key]["path"])) != record[key]:
            raise ValueError(f"An prepared/source file binding changed: {key}")
    if record["source"].get("sha256") != AN_SOURCE_SHA or record["input"].get("bytes") != INPUT_BYTES:
        raise ValueError("An preparation hash or tensor size mismatch")
    return record


def record_mac(destination: Path, device: str | None = None) -> None:
    import sounddevice as sd
    print("Nói trong 3 giây sau tín hiệu...", flush=True)
    time.sleep(1)
    print("THU ÂM", flush=True)
    samples = sd.rec(PCM_SAMPLES, samplerate=PCM_SAMPLE_RATE, channels=1,
                     dtype="int16", device=device, blocking=True)
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(PCM_SAMPLE_RATE)
        handle.writeframes(samples.astype("<i2").tobytes())


def _write_operation_report(out: Path, report: dict) -> None:
    write_json(out / "report.json", report)
    print("Report:", out / "report.json")


def provision_demo(args) -> None:
    if binding(MODEL)["sha256"] != MODEL_SHA:
        raise ValueError("Frozen model hash mismatch")
    records = load_prepared(args.prepared)
    enrollment = [record for record in records if record["role"] == "enroll"]
    out = new_output(args.output)
    encoder = None
    report = {"scope": "EXPLICIT_BOARD_PROVISION_TWO_FIXED_TEMPLATES",
              "backend": "usb", "device": None, "model": binding(MODEL),
              "prepared_manifest": binding(args.prepared / "manifest.json"),
              "enrollment_inputs": [item["input"] for item in enrollment],
              "board_role": "actual_encoder_derive_then_atomic_nvs_blob",
              "mac_role": "frontend_quantization_only",
              "auto_provision": False}
    try:
        encoder = USBEncoder(args.serial, out)
        report["device"] = encoder.device
        before = encoder.status()
        report["status_before"] = before
        if before.get("persistence_state") != "EMPTY":
            raise ValueError("Refusing provision: board is not empty; no overwrite performed")
        report["provision"] = encoder.provision(enrollment)
        report["status_after"] = encoder.status()
        report["status"] = "COMPLETED"
        _write_operation_report(out, report)
    except BaseException as error:
        report["status"] = "ERROR"
        report["error"] = str(error)
        _write_operation_report(out, report)
        raise
    finally:
        if encoder is not None:
            encoder.close()


def add_an_demo(args) -> None:
    if binding(MODEL)["sha256"] != MODEL_SHA:
        raise ValueError("Frozen model hash mismatch")
    enrollment = load_prepared_an(args.prepared_an)
    out = new_output(args.output)
    encoder = None
    report = {"scope": "EXPLICIT_APPEND_AN_TO_EXISTING_TWO_TEMPLATE_BOARD",
              "backend": "usb", "device": None, "model": binding(MODEL),
              "an_prepared_manifest": binding(args.prepared_an / "manifest.json"),
              "an_enrollment_input": enrollment["input"], "auto_provision": False}
    try:
        encoder = USBEncoder(args.serial, out)
        report["device"] = encoder.device
        before = encoder.status()
        report["status_before"] = before
        if (before.get("persistence_state"), before.get("enrollment_count"),
                before.get("labels")) != ("READY", 2, ["Quang", "Tien"]):
            raise ValueError("Refusing ADD_AN: board is not exactly the legacy two-template record")
        report["add_an"] = encoder.add_an(enrollment)
        report["status_after"] = encoder.status()
        report["status"] = "COMPLETED"
        _write_operation_report(out, report)
    except BaseException as error:
        report["status"] = "ERROR"
        report["error"] = str(error)
        _write_operation_report(out, report)
        raise
    finally:
        if encoder is not None:
            encoder.close()


def status_demo(args) -> None:
    out = new_output(args.output)
    encoder = None
    report = {"scope": "BOARD_PERSISTED_TEMPLATE_STATUS_READ_ONLY",
              "backend": "usb", "device": None, "auto_provision": False}
    try:
        encoder = USBEncoder(args.serial, out)
        report["device"] = encoder.device
        report["board_status"] = encoder.status()
        report["status"] = "COMPLETED"
        _write_operation_report(out, report)
    except BaseException as error:
        report["status"] = "ERROR"
        report["error"] = str(error)
        _write_operation_report(out, report)
        raise
    finally:
        if encoder is not None:
            encoder.close()


def board_result(item: dict, raw: bytes, scores: list[dict], reply: dict) -> dict:
    top1 = reply["top1"]
    return {"file": Path(item["source"]["path"]).name,
            "expected_label": item["label"], "top1": top1,
            "scores": scores, "raw_hex": raw.hex(), "response": reply,
            "correct": top1 == item["label"]}


def run_demo(args) -> None:
    if binding(MODEL)["sha256"] != MODEL_SHA:
        raise ValueError("Frozen model hash mismatch")
    if args.live and args.backend != "usb":
        raise ValueError("Live demo requires actual board backend")
    # A normal live query is deliberately independent of the frozen prepared
    # directory: enrollment stays on the board and is never reconstructed on
    # the Mac. The optional oracle explicitly opts back into prepared inputs.
    if args.live and not args.validate_mac:
        records = []
        queries = []
        enrollment = []
        prepared_manifest = None
    else:
        records = load_prepared(args.prepared)
        queries = [record for record in records if record["role"] == "query"]
        enrollment = [record for record in records if record["role"] == "enroll"]
        prepared_manifest = binding(args.prepared / "manifest.json")
    out = new_output(args.output)
    encoder = None
    report = {"scope": "CLOSED_SET_FIXED_QUERIES_OR_LIVE_DEMO_NOT_ACCURACY_BENCHMARK",
              "backend": args.backend, "device": None,
              "model": binding(MODEL),
              "prepared_manifest": prepared_manifest,
              "enrollment": [], "queries": [],
              "unknown_rejection": "CLOSED_SET_NO_UNKNOWN_THRESHOLD_OR_CLAIM",
              "matching": ("BOARD_TOP1_COSINE_ACTIVE_NVS_TEMPLATES"
                            if args.backend == "usb"
                            else "HOST_ONLY_TOP1_COSINE_FIXED_QUANG_TIEN"),
              "mac_role": ("frontend_quantization_optional_validation_oracle_only"
                            if args.backend == "usb"
                            else "frontend_quantization_matching_host_only"),
              "board_role": ("actual_encoder_boot_loaded_nvs_templates_cosine_matching"
                             if args.backend == "usb" else "NONE_HOST_ONLY"),
              "auto_provision": False}
    if args.live:
        report["mac_role"] = "PCM16_16KHZ_ACQUISITION_TRANSPORT_DISPLAY"
        report["board_role"] = "FRONTEND_INT8_ENCODER_NVS_COSINE"
    try:
        encoder = USBEncoder(args.serial, out) if args.backend == "usb" else HostEncoder(out)
        report["device"] = encoder.device
        report["fixture_parity"] = parity(encoder)
        mac_templates = None
        if args.backend == "usb":
            board_status = encoder.status()
            report["board_status"] = board_status
            if board_status.get("persistence_state") != "READY":
                raise ValueError("Board has no ready persisted templates; run explicit provision first")
            if args.validate_mac:
                mac_templates = {}
                for item in enrollment:
                    raw, reply = encoder.infer(Path(item["input"]["path"]).read_bytes())
                    mac_templates[item["label"]] = enroll(embedding(raw)[None, :])
                    report["enrollment"].append({"label": item["label"],
                                                  "input": item["input"],
                                                  "raw_hex": raw.hex(),
                                                  "response": reply,
                                                  "purpose": "OPTIONAL_MAC_ORACLE_ONLY"})
        else:
            templates = {}
            for item in enrollment:
                raw, reply = encoder.infer(Path(item["input"]["path"]).read_bytes())
                templates[item["label"]] = enroll(embedding(raw)[None, :])
                report["enrollment"].append({"label": item["label"], "input": item["input"],
                                              "raw_hex": raw.hex(), "response": reply})
        for item in queries:
            payload = Path(item["input"]["path"]).read_bytes()
            if args.backend == "usb":
                raw, scores, reply = encoder.recognize(payload)
                result = board_result(item, raw, scores, reply)
                if mac_templates is not None:
                    oracle_scores = rank(embedding(raw), mac_templates)
                    result["mac_validation_oracle"] = {
                        "top1": oracle_scores[0]["label"],
                        "scores": oracle_scores,
                        "agrees_with_board": oracle_scores[0]["label"] == result["top1"]}
            else:
                raw, reply = encoder.infer(payload)
                scores = rank(embedding(raw), templates)
                result = {"file": Path(item["source"]["path"]).name,
                          "expected_label": item["label"], "top1": scores[0]["label"],
                          "scores": scores, "raw_hex": raw.hex(), "response": reply,
                          "correct": scores[0]["label"] == item["label"]}
            report["queries"].append(result)
            print(result["file"], "→", result["top1"], result["scores"], flush=True)
        if args.live:
            wav16 = out / "live-raw-16k.wav"
            record_mac(wav16, args.device)
            payload = read_pcm_capture(wav16)
            raw, scores, reply = encoder.recognize_pcm(payload)
            report["live"] = {"wav": binding(wav16),
                              "preprocessing": "BOARD_FRONTEND_PCM16; no application DSP on Mac",
                              "scores": scores, "top1": reply["top1"],
                              "raw_hex": raw.hex(), "response": reply}
            if mac_templates is not None:
                oracle_scores = rank(embedding(raw), mac_templates)
                report["live"]["mac_validation_oracle"] = {
                    "top1": oracle_scores[0]["label"], "scores": oracle_scores,
                    "agrees_with_board": oracle_scores[0]["label"] == reply["top1"]}
            print("LIVE →", reply["top1"], scores, flush=True)
        report["status"] = "COMPLETED"
        _write_operation_report(out, report)
    except BaseException as error:
        report["status"] = "ERROR"
        report["error"] = str(error)
        _write_operation_report(out, report)
        raise
    finally:
        if encoder is not None:
            encoder.close()


def run_pcm_file(args) -> None:
    """Send an existing exact-format raw capture, without any host DSP."""
    payload = read_pcm_capture(args.wav)
    source = binding(args.wav)
    out = new_output(args.output)
    client = None
    report = {"scope": "PCM_BOARD_FRONTEND_CLOSED_SET_NOT_ACCURACY_BENCHMARK",
              "capture": source, "enrollment": [], "auto_provision": False,
              "mac_role": "PCM_CONTAINER_READ_TRANSPORT_DISPLAY",
              "board_role": "FRONTEND_INT8_ENCODER_NVS_COSINE"}
    try:
        client = USBEncoder(args.serial, out)
        report["device"] = client.device
        before = client.status()
        report["board_status"] = before
        if before.get("persistence_state") != "READY":
            raise ValueError("Ready persisted templates required; never auto-provision")
        started = time.perf_counter_ns()
        raw, scores, reply = client.recognize_pcm(payload)
        report.update(response=reply, raw_hex=raw.hex(), scores=scores,
                      top1=reply["top1"],
                      usb_roundtrip_us=(time.perf_counter_ns() - started) // 1000)
        after = client.status()
        report["board_status_after"] = after
        if (after.get("record_fingerprint"), after.get("labels")) != (
                before.get("record_fingerprint"), before.get("labels")):
            raise ValueError("Persisted templates changed during read-only PCM query")
        if binding(args.wav) != source:
            raise ValueError("PCM capture changed during run")
        report["status"] = "COMPLETED"
        _write_operation_report(out, report)
    except BaseException as error:
        report["status"] = "ERROR"
        report["error"] = str(error)
        _write_operation_report(out, report)
        raise
    finally:
        if client is not None:
            client.close()


def append_pcm_demo(args) -> None:
    label_bytes = safe_label(args.label)
    pcm = read_pcm_capture(args.wav)
    source = binding(args.wav)
    out = new_output(args.output)
    client = None
    report = {"scope": "EXPLICIT_APPEND_PCM_BOARD_ENROLLMENT", "label": args.label,
              "capture": source, "backend": "usb", "auto_provision": False}
    try:
        client = USBEncoder(args.serial, out)
        before = client.status()
        report["status_before"] = before
        if before.get("persistence_state") != "READY":
            raise ValueError("Ready persisted templates required; never auto-provision")
        before_labels = before.get("labels")
        if not _valid_active_labels(before_labels, before.get("enrollment_count")):
            raise ValueError("Board status labels are invalid")
        if args.label in before_labels or len(before_labels) >= MAX_TEMPLATES:
            raise ValueError("Refusing duplicate label or full template set")
        report["append_pcm"] = client.append_pcm(args.label, pcm)
        after = client.status()
        report["status_after"] = after
        if after.get("labels") != before_labels + [args.label]:
            raise ValueError("APPEND_PCM did not preserve the label prefix")
        if binding(args.wav) != source or safe_label(args.label) != label_bytes:
            raise ValueError("Enrollment source changed during operation")
        report["status"] = "COMPLETED"
        _write_operation_report(out, report)
    except BaseException as error:
        report.update(status="ERROR", error=str(error))
        _write_operation_report(out, report)
        raise
    finally:
        if client is not None:
            client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--data", type=Path, default=ROOT / "data")
    prep.add_argument("--output", type=Path, default=DEFAULT_PREPARED)
    prep_an = commands.add_parser("prepare-an")
    prep_an.add_argument("--data", type=Path, default=ROOT / "data")
    prep_an.add_argument("--output", type=Path, default=DEFAULT_AN_PREPARED)
    provision = commands.add_parser("provision")
    provision.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    provision.add_argument("--output", type=Path, required=True)
    provision.add_argument("--serial", default=SERIAL)
    add_an = commands.add_parser("add-an")
    add_an.add_argument("--prepared-an", type=Path, default=DEFAULT_AN_PREPARED)
    add_an.add_argument("--output", type=Path, required=True)
    add_an.add_argument("--serial", default=SERIAL)
    status = commands.add_parser("status")
    status.add_argument("--output", type=Path, required=True)
    status.add_argument("--serial", default=SERIAL)
    pcm_run = commands.add_parser("run-pcm", help="Raw exact-format WAV; board frontend")
    pcm_run.add_argument("--wav", type=Path, required=True)
    pcm_run.add_argument("--output", type=Path, required=True)
    pcm_run.add_argument("--serial", default=SERIAL)
    append_pcm = commands.add_parser("append-pcm", help="Explicit raw PCM enrollment; no host DSP")
    append_pcm.add_argument("--label", required=True)
    append_pcm.add_argument("--wav", type=Path, required=True)
    append_pcm.add_argument("--output", type=Path, required=True)
    append_pcm.add_argument("--serial", default=SERIAL)
    run = commands.add_parser("run")
    run.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--backend", choices=("usb", "host"), default="usb")
    run.add_argument("--serial", default=SERIAL)
    run.add_argument("--live", action="store_true")
    run.add_argument("--validate-mac", action="store_true",
                     help="Optional Mac score oracle; board remains the decision")
    run.add_argument("--device", help="Optional PortAudio microphone device name")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.data, args.output)
    elif args.command == "prepare-an":
        prepare_an(args.data, args.output)
    elif args.command == "provision":
        provision_demo(args)
    elif args.command == "add-an":
        add_an_demo(args)
    elif args.command == "status":
        status_demo(args)
    elif args.command == "run-pcm":
        run_pcm_file(args)
    elif args.command == "append-pcm":
        append_pcm_demo(args)
    else:
        run_demo(args)


if __name__ == "__main__":
    main()
