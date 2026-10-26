"""Host guards only; these tests do not establish board behavior."""
import importlib.util
import json
from pathlib import Path
import struct
import zlib

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "demo_voice_usb", Path(__file__).resolve().parents[1] / "scripts/demo_voice_usb.py")
DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEMO)


def reply():
    return dict(protocol_version=1, request_id=4, status="OK", operation="ENCODE",
                model_sha256=DEMO.MODEL_SHA, runtime_mode="ESP_NN_ANSI_C_REFERENCE",
                input_bytes=11920, output_bytes=192, input_scale=DEMO.INPUT_SCALE,
                input_zero_point=-12, output_scale=DEMO.OUTPUT_SCALE,
                output_zero_point=6, arena_capacity_bytes=131072,
                arena_used_bytes=38140, input_crc32=zlib.crc32(bytes(11920)),
                output_hex=bytes(192).hex(), invoke_us=53500)


def test_wire_header_and_crc():
    payload = bytes(11920)
    packet = DEMO.frame(4, payload)
    assert struct.unpack("<8sIII", packet[:20]) == (
        b"EDGEAI01", 4, 11920, zlib.crc32(payload))
    assert packet[20:] == payload
    assert DEMO.frame(5, b"") == struct.pack("<8sIII", b"EDGEAI01", 5, 0, 0)
    with pytest.raises(ValueError):
        DEMO.frame(0, b"bad")


def test_versioned_operation_payloads_are_bounded_and_distinct():
    tensor = bytes(DEMO.INPUT_BYTES)
    assert DEMO.operation_payload(DEMO.OP_ENCODE, tensor)[:2] == bytes((DEMO.OP_ENCODE, 1))
    assert len(DEMO.operation_payload(DEMO.OP_RECOGNIZE, tensor)) == DEMO.RECOGNIZE_PAYLOAD_BYTES
    assert len(DEMO.operation_payload(DEMO.OP_PROVISION, tensor + tensor)) == DEMO.PROVISION_PAYLOAD_BYTES
    assert len(DEMO.operation_payload(DEMO.OP_ADD_AN, tensor)) == DEMO.ADD_AN_PAYLOAD_BYTES
    assert DEMO.operation_payload(DEMO.OP_STATUS) == bytes((DEMO.OP_STATUS, 1))
    with pytest.raises(ValueError):
        DEMO.operation_payload(DEMO.OP_PROVISION, tensor)
    with pytest.raises(ValueError):
        DEMO.operation_payload(0x7F)


@pytest.mark.parametrize("key,value", [
    ("model_sha256", "0" * 64), ("runtime_mode", "OPTIMIZED"),
    ("input_scale", .5), ("input_zero_point", 0),
    ("output_bytes", 191), ("arena_used_bytes", 131073),
    ("protocol_version", True), ("output_scale", float("nan")),
])
def test_reject_contract_drift(key, value):
    record = reply()
    record[key] = value
    with pytest.raises(ValueError):
        DEMO.response_output(record, 4, bytes(11920))


@pytest.mark.parametrize("key,value", [
    ("request_id", 3), ("input_crc32", 0), ("status", "ERROR"),
    ("output_hex", "00"), ("invoke_us", 0),
])
def test_reject_invalid_response(key, value):
    record = reply()
    record[key] = value
    with pytest.raises(ValueError):
        DEMO.response_output(record, 4, bytes(11920))


def test_correct_response_and_embedding():
    assert DEMO.response_output(reply(), 4, bytes(11920)) == bytes(192)
    assert np.linalg.norm(DEMO.embedding(bytes(192))) == pytest.approx(1, abs=1e-6)
    with pytest.raises(ValueError):
        DEMO.embedding(bytes(191))


def test_board_recognition_response_requires_complete_active_set_and_board_ranked():
    record = dict(reply(), operation="RECOGNIZE", top1="Tien",
                  scores=[{"label": "Tien", "cosine": .8},
                          {"label": "Quang", "cosine": .7}])
    raw, scores = DEMO.recognition_output(record, 4, bytes(DEMO.INPUT_BYTES))
    assert raw == bytes(DEMO.OUTPUT_BYTES)
    assert scores[0]["label"] == "Tien"
    with pytest.raises(ValueError):
        DEMO.recognition_output(dict(record, top1="Quang"), 4,
                                bytes(DEMO.INPUT_BYTES))
    with pytest.raises(ValueError):
        DEMO.recognition_output(dict(record, scores=[{"label": "Other", "cosine": .8},
                                                     {"label": "Tien", "cosine": .7}]),
                                4, bytes(DEMO.INPUT_BYTES))
    an_record = dict(reply(), operation="RECOGNIZE", top1="An",
                     scores=[{"label": "An", "cosine": .8},
                             {"label": "Tien", "cosine": .7},
                             {"label": "Quang", "cosine": .6}])
    assert DEMO.recognition_output(an_record, 4, bytes(DEMO.INPUT_BYTES))[1][0]["label"] == "An"


def test_quantization_shape_finiteness_and_bytes():
    assert DEMO.quantize(np.zeros((298, 40), np.float32)) == np.full(11920, -12, np.int8).tobytes()
    with pytest.raises(ValueError):
        DEMO.quantize(np.zeros((1, 298, 40)))
    with pytest.raises(ValueError):
        DEMO.quantize(np.full((298, 40), np.nan))


def test_five_scores_follow_cosine_rank_not_registration_order():
    labels = ["LS700", "An", "Tien", "LS116", "Quang"]
    scores = [{"label": label, "cosine": .9 - i * .1} for i, label in enumerate(labels)]
    record = dict(reply(), operation="RECOGNIZE", top1="LS700", scores=scores)
    expected = ["Quang", "Tien", "An", "LS116", "LS700"]
    assert DEMO.recognition_output(record, 4, bytes(DEMO.INPUT_BYTES),
                                   expected_labels=expected)[1] == scores
    for invalid in ([None] + scores[1:],
                    [dict(scores[0], label="An")] + scores[1:],
                    [dict(scores[0], label="Bad Name")] + scores[1:],
                    [dict(scores[0], label="LS999")] + scores[1:],
                    [dict(scores[0], cosine=.1)] + scores[1:]):
        with pytest.raises(ValueError):
            DEMO.recognition_output(dict(record, scores=invalid), 4,
                                    bytes(DEMO.INPUT_BYTES), expected_labels=expected)


def test_prepared_split_cannot_enroll_query(tmp_path, monkeypatch):
    records = [dict(label=label, role=role, source={"path": filename})
               for label, role, filename in DEMO.FILES]
    records[-1]["role"] = "enroll"
    (tmp_path / "manifest.json").write_text(json.dumps(dict(
        model_sha256=DEMO.MODEL_SHA, query_used_for_enrollment=False, records=records)))
    monkeypatch.setattr(DEMO, "VOICE_MANIFEST_SHA",
                        DEMO.binding(tmp_path / "manifest.json")["sha256"])
    with pytest.raises(ValueError, match="split"):
        DEMO.load_prepared(tmp_path)


def test_prepared_rejects_modified_manifest_before_self_reported_bindings(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(dict(
        model_sha256=DEMO.MODEL_SHA, query_used_for_enrollment=False, records=[])))
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        DEMO.load_prepared(tmp_path)


@pytest.mark.skipif(
    not (DEMO.DEFAULT_PREPARED / "manifest.json").is_file(),
    reason="Private frozen voice preparation is not published",
)
def test_actual_independently_reviewed_voice_preparation_is_pinned():
    assert len(DEMO.load_prepared(DEMO.DEFAULT_PREPARED)) == 4


def test_new_output_refuses_overwrite(tmp_path):
    with pytest.raises(FileExistsError):
        DEMO.new_output(tmp_path)


def test_ranking_is_closed_set_without_acceptance_threshold():
    scores = DEMO.rank(np.array([1., 0.]), {
        "Tien": np.array([0., 1.]), "Quang": np.array([1., 0.])})
    assert scores == [{"label": "Quang", "cosine": 1.}, {"label": "Tien", "cosine": 0.}]


def test_usb_client_reassembles_chunks_and_ignores_stale_id(tmp_path):
    current = reply()
    stale = dict(current, request_id=3)
    wire = (b"boot log\n" + DEMO.PREFIX + json.dumps(stale).encode() + b"\n"
            + DEMO.PREFIX + json.dumps(current).encode() + b"\n")

    class Port:
        def __init__(self):
            self.writes, self.offset = [], 0

        def write(self, data):
            self.writes.append(data)
            return len(data)

        def flush(self):
            pass

        def read(self, _size):
            chunk = wire[self.offset:self.offset+37]
            self.offset += len(chunk)
            return chunk

    client = object.__new__(DEMO.USBEncoder)
    client.counter = 3
    client.port = Port()
    with (tmp_path / "log").open("wb") as client.log:
        raw, received = client.infer(bytes(11920))
    assert raw == bytes(192)
    assert received["request_id"] == 4
    expected_payload = DEMO.operation_payload(DEMO.OP_ENCODE, bytes(11920))
    assert b"".join(client.port.writes) == DEMO.frame(4, expected_payload)
    assert max(map(len, client.port.writes)) <= 256


def test_usb_client_propagates_error(tmp_path):
    error = dict(reply(), status="ERROR", error="CRC_MISMATCH")

    class Port:
        def write(self, data):
            return len(data)

        def flush(self):
            pass

        def read(self, _size):
            return DEMO.PREFIX + json.dumps(error).encode() + b"\n"

    client = object.__new__(DEMO.USBEncoder)
    client.counter = 3
    client.port = Port()
    with (tmp_path / "log").open("wb") as client.log:
        with pytest.raises(ValueError, match="CRC_MISMATCH"):
            client.infer(bytes(11920))


def test_provision_is_explicit_fixed_order_and_recognition_is_separate(tmp_path):
    quang = tmp_path / "Quang.int8"
    tien = tmp_path / "Tien.int8"
    quang.write_bytes(bytes(DEMO.INPUT_BYTES))
    tien.write_bytes(bytes(DEMO.INPUT_BYTES))
    records = [
        {"label": "Quang", "role": "enroll", "input": {"path": str(quang)}},
        {"label": "Tien", "role": "enroll", "input": {"path": str(tien)}},
    ]
    sent = {}
    client = object.__new__(DEMO.USBEncoder)

    def request(payload):
        sent["payload"] = payload
        return dict(reply(), operation="PROVISION", persistence_state="READY",
                    persistence_source="NVS_BOOT_READONLY", enrollment_count=2,
                    record_bytes=1640, record_crc32=0x1234ABCD,
                    record_fingerprint="crc32:1234abcd", labels=["Quang", "Tien"])

    client.request = request
    result = client.provision(records)
    assert result["operation"] == "PROVISION"
    assert sent["payload"][:2] == bytes((DEMO.OP_PROVISION, DEMO.OPERATION_VERSION))
    assert sent["payload"][2:2 + DEMO.INPUT_BYTES] == bytes(DEMO.INPUT_BYTES)
    assert sent["payload"][2 + DEMO.INPUT_BYTES:] == bytes(DEMO.INPUT_BYTES)
    with pytest.raises(ValueError, match="exactly Quang/Tien"):
        client.provision([dict(records[0], label="Other"), records[1]])

    source = (Path(__file__).resolve().parents[1] / "scripts/demo_voice_usb.py").read_text()
    run_source = source[source.index("def run_demo"):source.index("def main")]
    assert ".provision(" not in run_source
    assert "if args.live and not args.validate_mac" in run_source


def test_status_requires_deterministic_persisted_record_fingerprint():
    client = object.__new__(DEMO.USBEncoder)

    def request(_payload):
        return dict(reply(), operation="STATUS", status="OK",
                    persistence_state="READY",
                    persistence_source="NVS_BOOT_READONLY",
                    enrollment_count=2, record_bytes=1640,
                    record_crc32=0x1234ABCD,
                    record_fingerprint="crc32:1234abcd", labels=["Quang", "Tien"])

    client.request = request
    status = client.status()
    assert status["record_fingerprint"] == "crc32:1234abcd"

    def malformed_request(_payload):
        return dict(request(b""), record_fingerprint="crc32:00000000")

    client.request = malformed_request
    with pytest.raises(ValueError, match="fingerprint"):
        client.status()


def test_an_manifest_and_append_are_independent_from_legacy_prepared(tmp_path):
    source = tmp_path / "An.m4a"
    source.write_bytes(b"an-source")
    wav = tmp_path / "An.wav"
    wav.write_bytes(b"wav")
    tensor = tmp_path / "An.int8"
    tensor.write_bytes(bytes(DEMO.INPUT_BYTES))
    record = {"label": "An", "role": "enroll", "source": DEMO.binding(source),
              "wav": DEMO.binding(wav), "input": DEMO.binding(tensor)}
    manifest = {"scope": "EXPLICIT_AN_ENROLLMENT_NOT_QUERY_OR_BENCHMARK",
                "model_sha256": DEMO.MODEL_SHA, "source_sha256": DEMO.binding(source)["sha256"],
                "preprocessing": DEMO.AN_PREPROCESSING,
                "query_used_for_enrollment": False, "records": [record]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(DEMO, "AN_SOURCE_SHA", DEMO.binding(source)["sha256"])
    try:
        loaded = DEMO.load_prepared_an(tmp_path)
        client = object.__new__(DEMO.USBEncoder)
        sent = {}
        client.request = lambda payload: sent.setdefault("payload", payload) or dict(
            reply(), operation="ADD_AN", persistence_state="READY",
            persistence_source="NVS_BOOT_READONLY", enrollment_count=3,
            record_bytes=2416, record_crc32=1, record_fingerprint="crc32:00000001",
            labels=["Quang", "Tien", "An"])
        # Replace lambda result after capturing because non-empty bytes are truthy.
        def request(payload):
            sent["payload"] = payload
            return dict(reply(), operation="ADD_AN", persistence_state="READY",
                        persistence_source="NVS_BOOT_READONLY", enrollment_count=3,
                        record_bytes=2416, record_crc32=1, record_fingerprint="crc32:00000001",
                        labels=["Quang", "Tien", "An"])
        client.request = request
        client.add_an(loaded)
        assert sent["payload"][:2] == bytes((DEMO.OP_ADD_AN, DEMO.OPERATION_VERSION))
    finally:
        monkeypatch.undo()
