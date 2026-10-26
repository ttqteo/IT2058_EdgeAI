#!/usr/bin/env python3
"""Read a specific board's smoke log and independently compare its raw outputs.

No reset, flash, network, GPIO, or model invocation is performed by this tool.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import serial
from serial.tools import list_ports

MODEL_SHA256 = "aa42a3140f375f3a2c93648893132b880dd3e9f4be4ca8f82cde09fc6c440727"
MANIFEST_SHA256 = "4af129472ed9740ade5e66c1a68800f389e6b4c7f7d4de2f7b80dd59a0102a4f"


def binding(path):
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def verify_log(log, generated):
    if binding(generated / "manifest.json")["sha256"] != MANIFEST_SHA256:
        raise ValueError("Fixture tree is not the independently reviewed snapshot")
    manifest = json.loads((generated / "manifest.json").read_text())
    if manifest["scope"] != "BOARD_ENCODER_SMOKE_ONLY":
        raise ValueError("Unexpected fixture scope")
    model = binding(generated / manifest["model"]["embedded_file"])
    if model["sha256"] != MODEL_SHA256 or model["sha256"] != manifest["model"]["expected_sha256"]:
        raise ValueError("Embedded model binding changed")
    if [item["fixture_number"] for item in manifest["fixtures"]] != list(range(5)):
        raise ValueError("Expected exactly five reviewed fixtures")
    clean = re.sub(r"\x1b\[[0-9;]*m", "", log)
    chunks = {}
    for name, offset, value in re.findall(
            r"raw192bytes fixture=(\w+) offset=(\d+) data=([0-9a-f]{32})", clean):
        key = (name, int(offset))
        raw = bytes.fromhex(value)
        if key in chunks and chunks[key] != raw:
            raise ValueError("Conflicting raw output chunks")
        chunks[key] = raw
    fixtures = []
    for item in manifest["fixtures"]:
        if item["split"] != "train":
            raise ValueError("Non-training fixture refused")
        expected_path = generated / item["expected_file"]
        expected = expected_path.read_bytes()
        if binding(expected_path)["sha256"] != item["expected"]["sha256"]:
            raise ValueError("Expected-output binding changed")
        input_binding = binding(generated / item["input_file"])
        if input_binding["sha256"] != item["input"]["sha256"]:
            raise ValueError("Input binding changed")
        name = "real_train_%03d" % item["fixture_number"]
        actual = b"".join(chunks.get((name, offset), b"")
                          for offset in range(0, 192, 16))
        difference = None
        rms = None
        if len(actual) == len(expected) == 192:
            signed = lambda value: value if value < 128 else value - 256
            delta = [signed(a) - signed(b) for a, b in zip(actual, expected)]
            difference = max(abs(value) for value in delta)
            rms = (sum(value * value for value in delta) / 192) ** 0.5
        timing = re.search(r"fixture=" + name + r" invoke_us=(\d+) "
                           r"max_absdiff_lsb=(\d+) cutoff=1 status=(PASS|FAIL)", clean)
        passed = (difference is not None and difference <= 1 and timing is not None
                  and int(timing[2]) == difference and timing[3] == "PASS")
        fixtures.append({"name": name, "window_id": item["window_id"],
                         "input": input_binding, "raw_output_hex": actual.hex(),
                         "output_bytes": len(actual), "max_absdiff_lsb": difference,
                         "rms_diff_lsb": rms,
                         "invoke_us": int(timing[1]) if timing else None,
                         "status": "PASS" if passed else "FAIL"})
    arena = re.search(r"arena location_start=(\S+) location_end=(\S+) "
                      r"internal=DRAM capacity_bytes=(\d+) used_bytes=(\d+) aligned16=yes", clean)
    arena_data = None if arena is None else {
        "location_start": arena[1], "location_end": arena[2],
        "internal": "DRAM", "capacity_bytes": int(arena[3]),
        "used_bytes": int(arena[4])}
    final_pass = ("BOARD_ENCODER_SMOKE_ONLY RESULT=PASS" in clean
                  and "BOARD_ENCODER_SMOKE_ONLY RESULT=FAIL" not in clean
                  and model["sha256"] in clean and len(fixtures) == 5
                  and all(item["status"] == "PASS" for item in fixtures)
                  and arena_data is not None
                  and 0 < arena_data["used_bytes"] <= arena_data["capacity_bytes"] == 131072)
    return {"scope": "BOARD_ENCODER_SMOKE_ONLY", "model": model,
            "manifest": binding(generated / "manifest.json"), "fixtures": fixtures,
            "arena": arena_data, "status": "PASS" if final_pass else "FAIL",
            "limitations": ["PRECOMPUTED_TRAIN_LOGMEL_ONLY", "NO_MICROPHONE",
                            "NO_END_TO_END_ACCURACY_OR_LATENCY_CLAIM"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True, help="Exact USB serial/MAC")
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--log", type=Path, help="Verify an existing log without serial access")
    args = parser.parse_args()
    if not 0 < args.timeout <= 600:
        parser.error("timeout must be in (0, 600]")
    wanted = re.sub(r"[^0-9a-z]", "", args.serial.lower())
    if not wanted:
        parser.error("USB serial must not be empty")
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    log_path = args.output / "serial.log"
    observed_devices = []
    if args.log:
        with log_path.open("xb") as destination:
            destination.write(args.log.read_bytes())
    else:
        end = time.monotonic() + args.timeout
        accumulated = bytearray()
        with log_path.open("xb") as destination:
            while time.monotonic() < end:
                ports = [port for port in list_ports.comports()
                         if port.vid == 0x303A and port.pid == 0x1001
                         and re.sub(r"[^0-9a-z]", "", (port.serial_number or "").lower()) == wanted]
                if len(ports) > 1:
                    raise RuntimeError("Ambiguous USB device identity")
                if not ports:
                    time.sleep(0.2)
                    continue
                connection = serial.Serial(port=None, baudrate=115200, timeout=0.2)
                connection.dtr = False
                connection.rts = False
                connection.port = ports[0].device
                try:
                    connection.open()
                    observed_devices.append({"port": ports[0].device,
                                             "serial": ports[0].serial_number,
                                             "vid": ports[0].vid, "pid": ports[0].pid})
                    print("CAPTURE_READY port=" + ports[0].device, flush=True)
                    while time.monotonic() < end:
                        chunk = connection.read(4096)
                        if chunk:
                            destination.write(chunk)
                            destination.flush()
                            accumulated.extend(chunk)
                            print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
                        if re.search(rb"BOARD_ENCODER_SMOKE_ONLY RESULT=(PASS|FAIL)", accumulated):
                            end = 0
                            break
                except (serial.SerialException, OSError) as error:
                    print("Serial reconnect: " + str(error), flush=True)
                    time.sleep(0.2)
                finally:
                    if connection.is_open:
                        connection.close()
    result = verify_log(log_path.read_text(errors="replace"), args.generated)
    result["requested_serial"] = args.serial
    result["capture_mode"] = "OFFLINE_LOG_REPLAY" if args.log else "LIVE_USB_SERIAL"
    result["device_provenance"] = observed_devices
    if args.log:
        result["scope"] = "OFFLINE_LOG_CHECK_NOT_DEVICE_EVIDENCE"
    result["log"] = binding(log_path)
    with (args.output / "board_report.json").open("x") as destination:
        json.dump(result, destination, indent=2)
        destination.write("\n")
    print("INDEPENDENT_RAW_OUTPUT_CHECK=" + result["status"], flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
