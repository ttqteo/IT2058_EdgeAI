#!/usr/bin/env python3
"""Independent raw-tar/libsndfile coverage and mmap replay audit, no inference."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from edgeai.frontend import log_mel

EXPECTED = {"utterances": 28539, "speakers": 251, "windows": 135438,
            "source_frames": 5794034668, "padded_samples": 706989332}


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(root: Path, archive: Path) -> dict:
    if digest(archive, "md5") != "2a93770f6d5c6c964bc36631d331a522":
        raise ValueError("Official archive MD5 mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    rows = manifest["rows"]
    pcm = np.load(root / "pcm_int16.npy", mmap_mode="r", allow_pickle=False)
    features = np.load(root / "features.npy", mmap_mode="r", allow_pickle=False)
    assert pcm.dtype == np.int16 and pcm.shape == (135438, 48000)
    assert features.dtype == np.float32 and features.shape == (135438, 298, 40, 1)
    assert not pcm.flags.writeable and not features.flags.writeable
    assert len(rows) == EXPECTED["windows"]
    by_source = {}
    previous = None
    for index, row in enumerate(rows):
        key = (row["utterance_id"], row["start_sample"])
        assert previous is None or previous < key
        previous = key
        assert row["global_index"] == index
        assert row["window_id"] == f'{row["utterance_id"]}@{row["start_sample"]:010d}'
        by_source.setdefault(row["utterance_id"], []).append(index)
    tail = next(i for i, r in enumerate(rows) if r["valid_samples"] < 48000)
    probes = {0, len(rows) // 3, 2 * len(rows) // 3, len(rows) - 1, tail}
    wanted = {rows[i]["utterance_id"] for i in probes}
    seen, speakers, replays = set(), set(), []
    total_frames = total_padding = 0
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not member.name.endswith(".flac"):
                continue
            source_id = Path(member.name).stem
            assert source_id not in seen
            seen.add(source_id)
            payload = tar.extractfile(member).read()
            metadata = sf.info(io.BytesIO(payload))
            assert metadata.samplerate == 16000 and metadata.channels == 1
            assert metadata.subtype == "PCM_16"
            total_frames += metadata.frames
            source_hash = hashlib.sha256(payload).hexdigest()
            indices = by_source[source_id]
            assert len(indices) == (metadata.frames + 47999) // 48000
            decoded = sf.read(io.BytesIO(payload), dtype="int16")[0] if source_id in wanted else None
            for ordinal, index in enumerate(indices):
                row = rows[index]
                valid = min(48000, metadata.frames - ordinal * 48000)
                assert row["start_sample"] == ordinal * 48000
                assert row["input_samples"] == metadata.frames
                assert row["valid_samples"] == valid and row["padded_samples"] == 48000 - valid
                assert row["source_sha256"] == source_hash
                speaker, chapter, _ = source_id.split("-")
                assert str(row["speaker_id"]) == speaker and str(row["chapter_id"]) == chapter
                speakers.add(speaker)
                total_padding += 48000 - valid
                if valid < 48000:
                    assert not np.any(pcm[index, valid:])
                if index in probes:
                    expected = np.zeros(48000, dtype=np.int16)
                    expected[:valid] = decoded[ordinal * 48000:ordinal * 48000 + valid]
                    assert np.array_equal(pcm[index], expected)
                    recomputed = log_mel(expected.astype(np.float32) / 32768.0)
                    assert np.array_equal(features[index, :, :, 0], recomputed)
                    replays.append({"index": index, "window_id": row["window_id"], "pcm_exact": True,
                                    "features_exact": True, "valid_samples": valid})
    assert set(by_source) == seen
    counts = {"utterances": len(seen), "speakers": len(speakers), "windows": len(rows),
              "source_frames": total_frames, "padded_samples": total_padding}
    assert counts == EXPECTED, counts
    assert len(replays) == len(probes)
    zero_feature_rows = 0
    for start in range(0, len(rows), 256):
        chunk = features[start:start + 256]
        assert np.isfinite(chunk).all()
        zero_feature_rows += int(np.count_nonzero(~np.any(chunk != 0, axis=(1, 2, 3))))
    return {"status": "INDEPENDENT_DATA_AUDIT_PASS", "scope": "TRAIN only; no model inference",
            "counts": counts, "source_hours": total_frames / 16000 / 3600,
            "train_diagnostics": {"exact_zero_feature_rows": zero_feature_rows},
            "replays": sorted(replays, key=lambda item: item["index"]),
            "bindings": {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in
                         [archive, root / "manifest.json", root / "provenance.json",
                          root / "pcm_int16.npy", root / "features.npy", Path(__file__)]}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.archive)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "counts": result["counts"]}), flush=True)
