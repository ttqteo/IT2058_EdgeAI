"""Fail-closed retrospective metrics and pilot embedding evaluation.

This module implements the NEW_LOCAL_LIBRISPEECH_PILOT metric boundary.  The
retrospective threshold arithmetic follows SpeechBrain's ``metric_stats.py``
contract: target scores are accepted only when ``score > threshold`` and the
candidate set contains observed scores and adjacent midpoints.  The existing
application matcher is intentionally not used here; it keeps its separate
inclusive ``>=`` contract.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .frontend import NUM_SAMPLES
from .matching import cosine_similarity, enroll

_MANIFEST_FIELDS = {
    "utterance_id",
    "speaker_id",
    "chapter_id",
    "split",
    "wav_path",
    "wav_sha256",
    "source_sha256",
    "input_samples",
    "padded_samples",
    "truncated_samples",
}
_SPLITS = {
    "train",
    "validation",
    "calibration_enroll",
    "calibration_query",
    "test_enroll",
    "test_query",
}
_SPLIT_ALIASES = {"calibration": "calibration", "test": "test"}
_EMBEDDING_DIMENSION = 192


def _as_finite_vector(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real-valued numbers")
    try:
        vector = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _validate_metric_parameters(
    p_target: float, c_miss: float, c_fa: float
) -> tuple[float, float, float]:
    try:
        prior = float(p_target)
        miss_cost = float(c_miss)
        false_alarm_cost = float(c_fa)
    except (TypeError, ValueError) as exc:
        raise ValueError("metric parameters must be numeric") from exc
    if not np.isfinite(prior) or not 0.0 < prior < 1.0:
        raise ValueError("p_target must be finite and strictly between 0 and 1")
    if not np.isfinite(miss_cost) or miss_cost <= 0.0:
        raise ValueError("c_miss must be finite and positive")
    if not np.isfinite(false_alarm_cost) or false_alarm_cost <= 0.0:
        raise ValueError("c_fa must be finite and positive")
    return prior, miss_cost, false_alarm_cost


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    observed = np.unique(scores)
    if observed.size == 1:
        return observed.astype(np.float64, copy=False)
    midpoints = (observed[:-1] + observed[1:]) / 2.0
    # Every midpoint is between adjacent distinct finite values in the
    # normal cosine-score range.  Keeping the concatenated sorted sequence
    # mirrors the upstream candidate construction and preserves first-tie
    # selection.  Duplicate values, if float rounding creates one, have the
    # same masks and do not alter the selected threshold value.
    return np.sort(np.concatenate((observed, midpoints))).astype(
        np.float64, copy=False
    )


def evaluate_scores(
    labels: object,
    scores: object,
    *,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> dict[str, object]:
    """Evaluate binary cosine-like scores with the locked pilot arithmetic.

    ``labels`` uses ``1`` for target/same-speaker trials and ``0`` for
    non-target/different-speaker trials.  This function deliberately does not
    accept an empty class, silently drop invalid rows, or choose a threshold
    using an application ``>=`` matcher.
    """

    raw_labels = np.asarray(labels)
    if raw_labels.ndim != 1 or raw_labels.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if np.iscomplexobj(raw_labels):
        raise ValueError("labels must contain real-valued values")
    try:
        label_values = raw_labels.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must contain numeric 0/1 values") from exc
    if not np.all(np.isfinite(label_values)) or not np.all(
        (label_values == 0.0) | (label_values == 1.0)
    ):
        raise ValueError("labels must contain only binary values 0 and 1")

    score_values = _as_finite_vector(scores, "scores")
    if score_values.size != label_values.size:
        raise ValueError(
            "labels and scores must have the same number of entries; got "
            f"{label_values.size} and {score_values.size}"
        )

    prior, miss_cost, false_alarm_cost = _validate_metric_parameters(
        p_target, c_miss, c_fa
    )
    target_scores = score_values[label_values == 1.0]
    non_target_scores = score_values[label_values == 0.0]
    if target_scores.size == 0 or non_target_scores.size == 0:
        raise ValueError("labels must contain both target and non-target trials")

    thresholds = _candidate_thresholds(score_values)
    # The masks are intentionally explicit: this is the source-compatible
    # strict comparator, rather than sklearn's default >= ROC convention.
    misses = target_scores[:, None] <= thresholds[None, :]
    false_alarms = non_target_scores[:, None] > thresholds[None, :]
    frr = np.mean(misses, axis=0, dtype=np.float64)
    far = np.mean(false_alarms, axis=0, dtype=np.float64)

    eer_index = int(np.argmin(np.abs(far - frr)))
    eer = float((far[eer_index] + frr[eer_index]) / 2.0)

    detection_cost = (
        miss_cost * frr * prior
        + false_alarm_cost * far * (1.0 - prior)
    )
    dcf_index = int(np.argmin(detection_cost))
    min_dcf_raw = float(detection_cost[dcf_index])
    default_cost = min(
        miss_cost * prior, false_alarm_cost * (1.0 - prior)
    )

    return {
        "eer": eer,
        "eer_threshold": float(thresholds[eer_index]),
        "eer_far": float(far[eer_index]),
        "eer_frr": float(frr[eer_index]),
        "min_dcf_raw": min_dcf_raw,
        "min_dcf_normalized": float(min_dcf_raw / default_cost),
        "min_dcf_threshold": float(thresholds[dcf_index]),
        "target_trials": int(target_scores.size),
        "non_target_trials": int(non_target_scores.size),
        "p_target": prior,
        "c_miss": miss_cost,
        "c_fa": false_alarm_cost,
        "min_dcf_default_cost": float(default_cost),
        "candidate_thresholds": int(thresholds.size),
        "accept_comparator": "score > threshold",
        "target_miss_comparator": "target_score <= threshold",
        "false_alarm_comparator": "non_target_score > threshold",
        "threshold_method": "unique_scores_plus_adjacent_midpoints",
        "eer_method": "mean_far_frr_at_first_minimum_absolute_gap",
        "min_dcf_normalization": (
            "raw / min(c_miss*p_target, c_fa*(1-p_target))"
        ),
        "provenance": {
            "metric_arithmetic": "SpeechBrain metric_stats.py v1.0.3",
            "normalization_arithmetic": "NIST SRE16 section 3",
            "evaluation_status": "NEW_LOCAL_LIBRISPEECH_PILOT",
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a 64-character hexadecimal string")
    lowered = value.lower()
    if any(char not in "0123456789abcdef" for char in lowered):
        raise ValueError(f"{field} must be hexadecimal")
    return lowered


def _load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise ValueError("manifest format_version must be 1")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest rows must be a non-empty list")

    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not _MANIFEST_FIELDS <= set(row):
            raise ValueError(f"manifest row {index} is missing required fields")
        for field in ("utterance_id", "speaker_id", "chapter_id", "split", "wav_path"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"manifest row {index} has invalid {field}")
        utterance_id = row["utterance_id"]
        if utterance_id in seen_ids:
            raise ValueError(f"duplicate utterance_id {utterance_id!r}")
        seen_ids.add(utterance_id)
        if row["split"] not in _SPLITS:
            raise ValueError(f"manifest row {index} has unknown split")
        wav_path = Path(row["wav_path"])
        if wav_path.is_absolute() or ".." in wav_path.parts:
            raise ValueError(f"manifest row {index} has unsafe wav_path")
        _require_sha256(row["wav_sha256"], f"row {index} wav_sha256")
        _require_sha256(row["source_sha256"], f"row {index} source_sha256")
        for field in ("input_samples", "padded_samples", "truncated_samples"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"manifest row {index} has invalid {field}")
        if row["input_samples"] == 0:
            raise ValueError(f"manifest row {index} has no input samples")
        expected_padded = max(0, NUM_SAMPLES - row["input_samples"])
        expected_truncated = max(0, row["input_samples"] - NUM_SAMPLES)
        if (
            row["padded_samples"] != expected_padded
            or row["truncated_samples"] != expected_truncated
        ):
            raise ValueError(f"manifest row {index} has inconsistent crop metadata")
    return manifest, rows, hashlib.sha256(raw).hexdigest()


def _base_split(split: str) -> str:
    if split.endswith("_enroll"):
        return split.removesuffix("_enroll")
    if split.endswith("_query"):
        return split.removesuffix("_query")
    return split


def _validate_manifest_integrity(rows: list[dict[str, Any]], split: str) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]]
]:
    groups: dict[str, set[str]] = defaultdict(set)
    source_hashes: dict[str, str] = {}
    wav_hashes: dict[str, str] = {}
    for row in rows:
        group = _base_split(row["split"])
        groups[group].add(row["speaker_id"])
        for field, seen in (("source_sha256", source_hashes), ("wav_sha256", wav_hashes)):
            digest = row[field]
            previous = seen.get(digest)
            if previous is not None and previous != row["utterance_id"]:
                raise ValueError(f"duplicate {field} across utterances")
            seen[digest] = row["utterance_id"]

    group_names = sorted(groups)
    for left_index, left_name in enumerate(group_names):
        for right_name in group_names[left_index + 1 :]:
            overlap = groups[left_name] & groups[right_name]
            if overlap:
                raise ValueError(
                    "speaker leakage across splits: "
                    f"{left_name}/{right_name} share {sorted(overlap)!r}"
                )

    if split not in _SPLIT_ALIASES:
        raise ValueError("split must be 'calibration' or 'test'")
    enroll_name = f"{split}_enroll"
    query_name = f"{split}_query"
    enroll_rows = [row for row in rows if row["split"] == enroll_name]
    query_rows = [row for row in rows if row["split"] == query_name]
    if not enroll_rows or not query_rows:
        raise ValueError(f"manifest has no complete {split} enrollment/query split")

    enroll_speakers = {row["speaker_id"] for row in enroll_rows}
    query_speakers = {row["speaker_id"] for row in query_rows}
    if enroll_speakers != query_speakers:
        raise ValueError(f"{split} enrollment/query speaker sets differ")
    for speaker in sorted(enroll_speakers):
        speaker_enroll = [
            row for row in enroll_rows if row["speaker_id"] == speaker
        ]
        speaker_query = [row for row in query_rows if row["speaker_id"] == speaker]
        if len(speaker_enroll) != 3 or len(speaker_query) != 9:
            raise ValueError(
                f"{split} speaker {speaker!r} must have 3 enrollment and "
                f"9 query rows"
            )
    enroll_ids = {row["utterance_id"] for row in enroll_rows}
    query_ids = {row["utterance_id"] for row in query_rows}
    if enroll_ids & query_ids:
        raise ValueError(f"{split} enrollment/query reuse utterance IDs")
    enroll_sources = {row["source_sha256"] for row in enroll_rows}
    query_sources = {row["source_sha256"] for row in query_rows}
    enroll_wavs = {row["wav_sha256"] for row in enroll_rows}
    query_wavs = {row["wav_sha256"] for row in query_rows}
    if enroll_sources & query_sources or enroll_wavs & query_wavs:
        raise ValueError(f"{split} enrollment/query reuse PCM/file hashes")
    return enroll_rows, query_rows


def _load_embeddings(
    embeddings_path: Path,
    manifest_rows: list[dict[str, Any]],
    manifest_sha256: str,
) -> tuple[dict[str, np.ndarray], str, list[str]]:
    try:
        archive = np.load(embeddings_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read embeddings NPZ {embeddings_path}: {exc}") from exc
    with archive:
        required = {"ids", "embeddings", "manifest_sha256"}
        if not required <= set(archive.files):
            raise ValueError("embeddings NPZ is missing required arrays")
        ids = np.asarray(archive["ids"])
        values = np.asarray(archive["embeddings"])
        embedded_manifest = np.asarray(archive["manifest_sha256"])

    if ids.ndim != 1 or ids.dtype.kind != "U":
        raise ValueError("embeddings ids must be a one-dimensional Unicode array")
    if values.ndim != 2 or values.shape[1] != _EMBEDDING_DIMENSION:
        raise ValueError(
            "embeddings must have shape [N, 192] for the locked pilot model"
        )
    if values.dtype != np.dtype(np.float32):
        raise ValueError("embeddings must be float32")
    if values.shape[0] != len(manifest_rows) or ids.size != len(manifest_rows):
        raise ValueError("embeddings and manifest must contain the same rows")
    if embedded_manifest.ndim != 0 or embedded_manifest.dtype.kind != "U":
        raise ValueError("manifest_sha256 must be a Unicode scalar in the NPZ")
    if str(embedded_manifest.item()) != manifest_sha256:
        raise ValueError("embeddings NPZ manifest_sha256 does not match manifest")
    if len(set(ids.tolist())) != ids.size:
        raise ValueError("embeddings ids must be unique")
    manifest_ids = [row["utterance_id"] for row in manifest_rows]
    if ids.tolist() != manifest_ids:
        raise ValueError("embeddings ids must exactly match manifest row order")
    if not np.all(np.isfinite(values)):
        raise ValueError("embeddings must contain only finite values")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("embeddings must be finite and non-zero")
    if not np.allclose(norms, 1.0, rtol=2e-4, atol=2e-4):
        raise ValueError("embeddings must be L2-normalized")
    return (
        {identifier: values[index] for index, identifier in enumerate(manifest_ids)},
        _sha256_file(embeddings_path),
        manifest_ids,
    )


def evaluate_embeddings(
    manifest_path: str | Path,
    embeddings_path: str | Path,
    *,
    split: str = "test",
) -> dict[str, object]:
    """Evaluate all query-to-enrollment-template trials in a pilot split.

    The enrollment template is the normalized mean of exactly three
    normalized utterance embeddings.  This function is intentionally limited
    to the documented ``calibration`` and ``test`` split pairs and returns a
    JSON-serializable score table for auditability.
    """

    manifest_file = Path(manifest_path)
    embeddings_file = Path(embeddings_path)
    manifest, rows, manifest_sha256 = _load_manifest(manifest_file)
    enroll_rows, query_rows = _validate_manifest_integrity(rows, split)
    embedding_by_id, embeddings_sha256, _ = _load_embeddings(
        embeddings_file, rows, manifest_sha256
    )

    enrollment_by_speaker: dict[str, tuple[list[str], np.ndarray]] = {}
    for speaker in sorted({row["speaker_id"] for row in enroll_rows}):
        identifiers = [
            row["utterance_id"] for row in enroll_rows if row["speaker_id"] == speaker
        ]
        template = enroll(np.stack([embedding_by_id[item] for item in identifiers]))
        enrollment_by_speaker[speaker] = (identifiers, template)

    labels: list[int] = []
    scores: list[float] = []
    score_rows: list[dict[str, object]] = []
    for query_row in query_rows:
        query_embedding = embedding_by_id[query_row["utterance_id"]]
        for enrollment_speaker, (enrollment_ids, template) in enrollment_by_speaker.items():
            label = int(query_row["speaker_id"] == enrollment_speaker)
            score = cosine_similarity(query_embedding, template)
            labels.append(label)
            scores.append(score)
            score_rows.append(
                {
                    "query_id": query_row["utterance_id"],
                    "query_speaker_id": query_row["speaker_id"],
                    "enrollment_speaker_id": enrollment_speaker,
                    "enrollment_ids": list(enrollment_ids),
                    "label": label,
                    "score": float(score),
                }
            )

    result = evaluate_scores(labels, scores)
    result.update(
        {
            "status": "NEW_LOCAL_LIBRISPEECH_PILOT",
            "dataset": manifest.get("dataset", ""),
            "split": split,
            "manifest_sha256": manifest_sha256,
            "embeddings_sha256": embeddings_sha256,
            "embedding_shape": [len(rows), _EMBEDDING_DIMENSION],
            "enrollment_templates": len(enrollment_by_speaker),
            "query_utterances": len(query_rows),
            "score_trials": len(score_rows),
            "enrollment_embeddings_per_template": 3,
            "score_method": (
                "cosine(query_embedding, normalized_mean_of_three_enrollment_embeddings)"
            ),
            "trial_provenance": (
                "manifest speaker_id equality; NEW_LOCAL_LIBRISPEECH_PILOT; "
                "not VoxCeleb1-O"
            ),
            "manifest_path": str(manifest_file.resolve()),
            "embeddings_path": str(embeddings_file.resolve()),
            "scores": score_rows,
        }
    )
    return result


__all__ = ["evaluate_embeddings", "evaluate_scores"]
