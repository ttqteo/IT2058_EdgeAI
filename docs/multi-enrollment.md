# Multi-speaker enrollment

The board retains the existing `Quang`, `Tien`, `An` templates and may append
up to two public-demo speakers, for a hard maximum of five templates. No UI
action writes enrollment data; dashboard actions remain STATUS/query only.

`demo_voice_usb.py append-pcm --label LS116 --wav sample.wav --output out`
is the explicit write path. The WAV must be exactly mono PCM16, 16 kHz, and
48,000 frames (96,000 raw bytes). Labels are 1–7 ASCII letters/digits/`-`/`_`,
encoded as an 8-byte NUL-padded field. The client reads STATUS before and after
the write, rejects duplicate/full/unready state, checks HELLO capability, and
requires the prior label prefix to be preserved.

The v1 (2-template, 1640-byte) and v2 (3-template, 2416-byte) records are
read-only compatible. `APPEND_PCM` (`0x21`, operation version 1) carries:

```
0x21 | 0x01 | label[8] | pcm16le[96000]
```

The board computes the locked log-mel/quantization/encoder/normalization path,
then writes a v3 `EAI-TPL3` 3972-byte record with model binding and CRC32.
The record has 3–5 vectors and labels; slots 0–2 must remain `Quang`, `Tien`,
`An`. NVS publication occurs only after set, commit, exact readback, and
verification. Any uncertain persistence error fails closed until reboot.

Persistence buffers are heap-scoped. The boot parser retains two 3884-byte
`TemplateSet` objects simultaneously; the compile-time stack guard accounts
for both plus 8192 bytes of call/NVS margin (15960 bytes), below the configured
16384-byte main-task stack. The v3 writer accepts vector/label references and
does not create a second full candidate on the append call chain.

Score arrays are ranked by cosine, not registration order. The client requires
safe unique labels, descending finite cosine scores, and the exact active label
set from its checked STATUS snapshot. STATUS itself preserves registration order.

These are source/build contracts, not evidence that five templates have already
been deployed. Physical acceptance requires backups, app-only deployment,
explicit appends, independent preservation checks, and reboot/query evidence.
