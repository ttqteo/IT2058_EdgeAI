# PCM frontend local optimization

Status updated 2026-10-23: the candidate passed host checks and a bounded
real-board check of two fixed PCM queries, repeated twice. Independent cached
evidence review passed. This is not live-microphone or general-accuracy evidence.
The local-development sections below retain the original task's scope; the
deployment results are recorded separately at the end.

## Scope and preserved contract

The change is limited to `pcm_frontend.cc`, `pcm_frontend.h`, the new
frontend-only `pcm_frontend_tables.h`, the focused frontend test, and the
evidence directory for this date. `main.cc`, the template store, model,
client/UI, SDK/configuration, CMake/dependencies, slides, and board state were
not changed.

The external frontend contract remains:

- little-endian signed PCM16, mono, 16,000 Hz, exactly 48,000 samples;
- periodic Hann, frame 400, hop 160, 512-point unnormalized FFT;
- 40-bin float32 HTK mel coefficients, natural log with floor `1e-6`;
- per-mel CMN over 298 frames;
- runtime model scale/zero point, ties-to-even rounding, and int8 saturation.

The existing accepted template fingerprint `crc32:60a382e2` and the existing
board artifacts are outside this task and were not touched.

## Candidate

The candidate keeps double-precision FFT and accumulation because an attempted
float32 FFT variant exceeded the unchanged feature gate (up to about `1.7e-3`
absolute feature error on synthetic data). The accepted candidate instead:

1. precomputes the 400 periodic-Hann coefficients;
2. precomputes 256 double FFT twiddles and the 512-entry bit-reversal map;
3. replaces the dense 40x257 mel traversal with 494 nonzero entries derived
   from the locked filterbank;
4. computes each sparse mel power directly, removing the unused 257-bin power
   scratch array;
5. retains the existing in-place PCM feature store and quantization behavior.

No progress hook, watchdog setting, timeout, scheduling API, or shared API was
added. Lead can schedule frontend work separately if the board check still
needs it.

## Independent parity

The reference source was copied before edits to
`evidence/esp32-smoke/20261022/pcm-optimization-local/before/`; its manifest is
in that directory. The candidate was compiled separately from that snapshot
using clang with `-std=c++17 -O2 -Wall -Wextra -Werror -pedantic`.

Across five synthetic inputs and the first 48,000 samples of three prepared
real recordings, the candidate was bit-identical to the reference: max feature
difference `0`, RMS feature difference `0`, and zero int8 differences on every
case. Against the independent NumPy oracle, the largest max feature error was
`9.686e-7`, the largest RMS error was `2.828e-7`, and every quantized tensor
was exactly equal. Full per-case values are in
`evidence/esp32-smoke/20261022/pcm-optimization-local/parity-report.json`.

| Input | Max feature error vs oracle | RMS error vs oracle | Int8 differences |
| --- | ---: | ---: | ---: |
| silence | 6.395e-14 | 6.395e-14 | 0 |
| impulse | 9.537e-7 | 2.828e-7 | 0 |
| bin sine | 9.239e-7 | 2.425e-7 | 0 |
| deterministic noise | 4.768e-7 | 1.731e-7 | 0 |
| chirp/clipping | 9.537e-7 | 2.348e-7 | 0 |
| Quang_test | 9.537e-7 | 1.985e-7 | 0 |
| Tien_test | 9.537e-7 | 1.692e-7 | 0 |
| An | 9.686e-7 | 2.631e-7 | 0 |

## Resource and host evidence

| Item | Reference | Candidate |
| --- | ---: | ---: |
| `Workspace` bytes | 10,568 | 8,512 |
| logical mel/table payload | 41,120 B dense | 12,354 B sparse + constants |
| clang object `__TEXT` | 44,055 B | 14,351 B |

The host runner was repeated 100 times on `Quang_test.pcm`. Its end-to-end
macOS timing, including process startup and file I/O, was `1.00 s` reference
versus `0.44 s` candidate. This is indicative host evidence only; it is not an
ESP32 speedup claim. The timing record is retained in
`evidence/esp32-smoke/20261022/pcm-optimization-local/host-timing.txt`.

## Checks

Passed locally:

- `./.venv/bin/python -m pytest -q tests/test_pcm_frontend.py` — 18 passed;
- `./.venv/bin/python -m pytest -q tests/test_esp32_pcm.py tests/test_pcm_frontend.py` — 24 passed;
- `./.venv/bin/python -m pytest -q tests/test_frontend.py tests/test_c_frontend.py tests/test_esp32_pcm.py tests/test_pcm_frontend.py` — 48 passed.

Not run by the original scoped frontend task: ESP-IDF build, board flash/reset, serial capture,
physical timing, watchdog behavior, live microphone capture, and demo
enrollment. Lead must rerun the board build and hardware gates before accepting
the candidate. The prior approximately 7.7-second board frontend and its
watchdog warning were pending at that original handback. See the later bounded
board check below.

## Subsequent bounded board check

Lead built with the existing ESP-IDF 5.5.1 configuration and deployed only the
app at `0x10000`. Candidate binary SHA-256:
`6db473d56d9887e1b778292397cb2b648182a5de0448d2390ae885b66f23e325`.
Image size: 722,016 B; factory-slot remaining: 326,560 B.

Evidence directory:
`artifacts/esp32-smoke/20261022/pcm-optimization-candidate-001/`.
`preflash.json` and `build-verification.json` are historical pre-write/local
snapshots, not the final deployment state. Later `flash-app-only.log`, NVS and
partition readbacks, and `pcm-check-001/002` establish the bounded later state.

| Measured scope | Previous PCM build | Candidate run 001 |
| --- | ---: | ---: |
| Frontend — Quang / Tien | 7.659 / 7.656 s | 2.854 / 2.851 s |
| Board recognition — Quang / Tien | 7.715 / 7.712 s | 2.909 / 2.907 s |
| USB request/response — Quang / Tien | 8.101 / 8.099 s | 3.218 / 3.222 s |

The frontend reduction is about 62.7% (~2.68× speedup) on these fixed inputs.
The encoder Invoke remains about 53.6 ms; it is not the full PCM-pipeline time.
Run 002 repeated recognition at 2.909 / 2.907 s. Neither retained candidate
serial log contains a task-watchdog warning; the previous run contained two.
This does not prove watchdog-free behavior on arbitrary inputs or continuous use.

Both runs reproduced the legacy encoder output exactly for both PCM queries
(embedding difference 0 LSB), matched the frozen feature CRCs, and retained
five encoder parity maxima `[1, 0, 0, 0, 1]` LSB. Quang/Tien fixed-query top-1
labels were correct. No enrollment or live microphone operation was used.

NVS before/after SHA-256 remained
`c6d2c445312fa128329741cb82a49c882662ceae3c97fe69a1840794c324d80e`;
the partition readbacks were also identical. Active templates remained
Quang/Tien/An, fingerprint `crc32:60a382e2`. Later enrollment work has separate
artifacts and does not retroactively change this three-template result.

Workspace saving is 2,056 B (`10,568 → 8,512`). The raw recognition payload
plus workspace has a logical sum of 104,514 B, excluding allocator metadata,
stack and other allocations. Query heap delta is a before/after snapshot,
not peak-memory usage. Linker free DIRAM is not runtime free heap.
