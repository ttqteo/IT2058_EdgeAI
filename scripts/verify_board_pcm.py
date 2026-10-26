#!/usr/bin/env python3
"""Explicit offline oracle + real-board PCM integration check, never live DSP.

Only two frozen query WAVs are used. No enrollment, training, microphone,
reset or flash operation. Generated windows are diagnostic fixtures, not
evidence of raw microphone capture or a mic attached to the MCU.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
import time
import wave
import zlib

SPEC = importlib.util.spec_from_file_location(
    "demo_voice_usb", Path(__file__).with_name("demo_voice_usb.py"))
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


def run(output: Path) -> None:
    records = demo.load_prepared(demo.DEFAULT_PREPARED)
    queries = [item for item in records if item["role"] == "query"]
    out = demo.new_output(output)
    report = {"scope": "REAL_BOARD_PCM_VS_OFFLINE_ORACLE_TWO_FROZEN_QUERIES",
              "not_live_microphone_evidence": True, "enrollment": [],
              "auto_provision": False, "queries": []}
    client = None
    try:
        fixtures = []
        for item in queries:
            source = Path(item["wav"]["path"])
            with wave.open(str(source), "rb") as handle:
                if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (16000, 1, 2):
                    raise ValueError("Frozen WAV format changed")
                pcm = handle.readframes(demo.PCM_SAMPLES)
            padded_bytes = demo.PCM_BYTES - len(pcm)
            pcm += bytes(padded_bytes)
            fixture = out / (source.stem + "-oracle-3s.wav")
            with wave.open(str(fixture), "wb") as handle:
                handle.setparams((1, 2, 16000, demo.PCM_SAMPLES, "NONE", "NONE"))
                handle.writeframes(pcm)
            reference_pcm, _ = demo.load_wav(fixture)
            expected_tensor = demo.quantize(demo.log_mel(reference_pcm))
            if expected_tensor != Path(item["input"]["path"]).read_bytes():
                raise ValueError("Offline oracle differs from frozen tensor")
            fixtures.append((item, fixture, pcm, expected_tensor, padded_bytes))
        client = demo.USBEncoder(demo.SERIAL, out)
        report["device"] = client.device
        before = client.status()
        report["status_before"] = before
        if (before.get("persistence_state"), before.get("labels"), before.get("record_fingerprint")) != (
                "READY", ["Quang", "Tien", "An"], "crc32:60a382e2"):
            raise ValueError("Expected the accepted three-template NVS state")
        report["legacy_encoder_parity"] = demo.parity(client)
        for item, fixture, pcm, tensor, padded in fixtures:
            legacy_raw, legacy_scores, legacy_reply = client.recognize(tensor)
            started = time.perf_counter_ns()
            raw, scores, reply = client.recognize_pcm(pcm)
            roundtrip = (time.perf_counter_ns() - started) // 1000
            differences = [abs((a if a < 128 else a - 256) -
                               (b if b < 128 else b - 256))
                           for a, b in zip(raw, legacy_raw)]
            result = {"label": item["label"], "source": item["wav"],
                      "fixture": demo.binding(fixture), "offline_padding_bytes": padded,
                      "pcm_crc32": zlib.crc32(pcm), "feature_crc32_expected": zlib.crc32(tensor),
                      "response": reply, "legacy_response": legacy_reply,
                      "usb_roundtrip_us": roundtrip,
                      "embedding_max_absdiff_lsb": max(differences),
                      "feature_crc_matches": reply["feature_crc32"] == zlib.crc32(tensor),
                      "top1_matches_legacy": reply["top1"] == legacy_reply["top1"],
                      "correct": reply["top1"] == item["label"]}
            report["queries"].append(result)
            if not (result["feature_crc_matches"] and result["top1_matches_legacy"]
                    and result["correct"] and result["embedding_max_absdiff_lsb"] <= 1):
                raise ValueError("PCM integration gate failed; preserve the negative evidence")
        after = client.status()
        report["status_after"] = after
        for key in ("record_fingerprint", "record_crc32", "record_bytes", "labels", "enrollment_count"):
            if after.get(key) != before.get(key):
                raise ValueError("NVS status changed during read-only verification")
        report["status"] = "PASS_TWO_FIXED_QUERIES_ONLY"
    except BaseException as error:
        report["status"] = "ERROR"
        report["error"] = str(error)
        raise
    finally:
        if client is not None:
            client.close()
        demo.write_json(out / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
