# Board telemetry

This is the additive, read-only telemetry contract for `PERF-FW-01`. The
firmware emits an optional `telemetry` JSON object only on successful
`STATUS` and `RECOGNIZE` replies, with the PCM extension below. A client must continue to accept a reply
without this object; `scripts/demo_voice_usb.py::validate_telemetry(reply)`
returns `None` for that compatibility path.

## Fields

All byte counts are bytes. `version` is the numeric value `1`; every other
base field is a non-negative integer.

| Field | Meaning |
| --- | --- |
| `version` | Telemetry schema version, always `1`. |
| `uptime_us` | Board uptime in microseconds at the snapshot. |
| `internal_heap_total_bytes` | Total heap capacity matching `MALLOC_CAP_INTERNAL \| MALLOC_CAP_8BIT`. |
| `internal_heap_free_bytes` | Current free bytes for that capability set. |
| `internal_heap_min_free_bytes` | Lifetime low-water estimate aggregated across matching regions. |
| `internal_heap_largest_free_block_bytes` | Largest currently allocatable block for that capability set. |

`STATUS` contains the six base fields. Successful `RECOGNIZE` adds:

| Field | Meaning |
| --- | --- |
| `query_heap_free_before_bytes` | Free internal 8-bit heap immediately before `InvokeRequest`. |
| `query_heap_free_after_bytes` | Free internal 8-bit heap after normalization, cosine scoring, and stable top-1 sorting, before response formatting. |
| `query_heap_delta_bytes` | Signed `before - after`; this is not a peak or allocation total. |
| `recognition_us` | Board-side input-copy + invoke + normalization/cosine/top-1 work, excluding JSON formatting and USB transfer. It includes the existing top-level `invoke_us`. |

## PCM extension

The additive `RECOGNIZE_PCM` operation uses the same telemetry schema. Its
before/after snapshots surround board frontend, quantization, encoder, and
cosine work. `recognition_us` therefore includes frontend processing for this
operation, but excludes raw-input/feature CRC work, JSON and USB. The top-level
`frontend_us` measures frontend plus quantization; `invoke_us` remains encoder
Invoke only. The client requires `frontend_us + invoke_us <= recognition_us`.
Transport roundtrip is measured separately on the Mac and includes board
processing. It is not a substitute for either board timer.

PCM and frontend workspace are reused allocations. A zero heap delta does not
mean the query requires zero RAM; before/after heap snapshots do not measure
peak stack or peak temporary memory. The historical build figures below are
for PERF-FW-01, not the new PCM image.

The firmware uses the exact ESP-IDF 5.5.1 `heap_caps_get_*` APIs declared in
`artifacts/esp32-toolchain/esp-idf-v5.5.1/components/heap/include/esp_heap_caps.h`.
It does not start or reset a heap low-water monitor. The static tensor arena is
reusable persistent memory and is not presented as exact total RAM or exact
per-query memory usage.

The client rejects unknown versions, malformed or partial objects, booleans in
integer fields, impossible capacity/free relationships, inconsistent deltas,
and impossible recognition/invoke timings. The focused tests use synthetic
replies only and are not live-board evidence.

## Verification

The source-only firmware check is the repository-local IDF 5.5.1 build:

```sh
bash scripts/esp32_idf.sh -C firmware/esp32_s3_smoke build
bash scripts/esp32_idf.sh -C firmware/esp32_s3_smoke size
```

Before rebuilding, the prior app, ELF, and active `sdkconfig` were preserved
under `artifacts/esp32-smoke/20261020/perf-build-before-telemetry/`. The build
must retain the existing ANSI-C mode, model/fixture pins, response capacity,
stack guard, and frozen artifacts. A successful compile is not a board, USB,
reset, flash, or live telemetry claim.

The preserved pre-change app and ELF hashes are respectively
`b98f92f4331c532d29adba53871e493dd252f7484132885d00ec0f482830b94d` and
`a16e36684da615c38761d76940009fe172aa26a58468767241496a361553ec62`;
the preserved config hash is the same as the active config below.

For this source build, the app image is `704864` bytes (`0xac160`), leaving
`0x53ea0` bytes free in the `0x100000`-byte app partition. The IDF size tool
reports `704749` bytes of total image content before binary padding. The new
app SHA-256 is
`36e93554659c6602eba7a19eb9bdaf087af31b8a7bce30977b90aa8dd7862ef1`; the ELF
SHA-256 is
`fd7c330c8ab10b069b443978dbd914082b6b1726e3aa5c2514655cf1d69e64a8`; the
active `sdkconfig` SHA-256 remains
`d530251dc71a63dba37afe81e2ac42a33c58f0fe22a08527d99030a8729e9056`.

The synthetic maximum-size `RECOGNIZE` envelope calculation is 1,419 bytes
including the `EDGEAI_RESPONSE ` prefix and newline, below the unchanged
1,536-byte response buffer. This is a serialization bound check, not live USB
evidence.
