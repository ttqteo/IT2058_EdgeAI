from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from edgeai.evaluation import evaluate_embeddings, evaluate_scores


def _independent_bruteforce(
    labels: list[int], scores: list[float], p_target: float = 0.01
) -> dict[str, float]:
    """Small scalar oracle deliberately independent of the vectorized code."""

    observed = sorted(set(scores))
    thresholds = sorted(observed + [
        (left + right) / 2.0
        for left, right in zip(observed, observed[1:])
    ])
    target = [score for label, score in zip(labels, scores) if label == 1]
    non_target = [score for label, score in zip(labels, scores) if label == 0]
    operating_points: list[tuple[float, float, float, float]] = []
    for index, threshold in enumerate(thresholds):
        frr = sum(score <= threshold for score in target) / len(target)
        far = sum(score > threshold for score in non_target) / len(non_target)
        dcf = p_target * frr + (1.0 - p_target) * far
        operating_points.append((threshold, frr, far, dcf))
    eer_index = min(
        range(len(operating_points)),
        key=lambda index: (
            abs(operating_points[index][1] - operating_points[index][2]),
            index,
        ),
    )
    dcf_index = min(
        range(len(operating_points)),
        key=lambda index: (operating_points[index][3], index),
    )
    eer_point = operating_points[eer_index]
    dcf_point = operating_points[dcf_index]
    return {
        "eer": (eer_point[1] + eer_point[2]) / 2.0,
        "eer_threshold": eer_point[0],
        "min_dcf_raw": dcf_point[3],
        "min_dcf_threshold": dcf_point[0],
    }


def test_scores_match_independent_oracle_and_strict_tie_selection() -> None:
    labels = [1, 1, 0, 0]
    scores = [0.6, 0.4, 0.5, 0.3]
    expected = _independent_bruteforce(labels, scores)
    result = evaluate_scores(labels, scores)

    for key, value in expected.items():
        assert result[key] == pytest.approx(value)
    assert result["eer"] == pytest.approx(0.5)
    assert result["eer_threshold"] == pytest.approx(0.4)
    assert result["min_dcf_raw"] == pytest.approx(0.005)
    assert result["min_dcf_normalized"] == pytest.approx(0.5)
    assert result["min_dcf_threshold"] == pytest.approx(0.5)
    assert result["accept_comparator"] == "score > threshold"
    assert result["target_trials"] == 2
    assert result["non_target_trials"] == 2
    json.dumps(result)


def test_perfect_separation_uses_first_zero_operating_point() -> None:
    result = evaluate_scores([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert result["eer"] == pytest.approx(0.0)
    # With strict >, a negative score equal to .2 is not a false alarm.
    assert result["eer_threshold"] == pytest.approx(0.2)
    assert result["min_dcf_raw"] == pytest.approx(0.0)
    assert result["min_dcf_normalized"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("labels", "scores"),
    [
        ([1, 0], [0.1]),
        ([1, 1], [0.1, 0.2]),
        ([1, 2], [0.1, 0.2]),
        ([1, 0], [np.nan, 0.2]),
        ([1, 0], [np.inf, 0.2]),
    ],
)
def test_score_input_rejection(labels: object, scores: object) -> None:
    with pytest.raises(ValueError):
        evaluate_scores(labels, scores)


@pytest.mark.parametrize(
    ("p_target", "c_miss", "c_fa"),
    [(0.0, 1.0, 1.0), (1.0, 1.0, 1.0), (0.01, 0.0, 1.0), (0.01, 1.0, -1.0)],
)
def test_metric_parameter_rejection(
    p_target: float, c_miss: float, c_fa: float
) -> None:
    with pytest.raises(ValueError):
        evaluate_scores(
            [1, 0], [0.9, 0.1], p_target=p_target, c_miss=c_miss, c_fa=c_fa
        )


def _write_embedding_fixture(tmp_path):
    rows: list[dict[str, object]] = []
    values: list[np.ndarray] = []
    identifiers: list[str] = []
    for speaker, dimension in (("speaker-a", 0), ("speaker-b", 1)):
        vector = np.zeros(192, dtype=np.float32)
        vector[dimension] = 1.0
        for role, count in (("enroll", 3), ("query", 9)):
            split = f"test_{role}"
            for index in range(count):
                identifier = f"{speaker}-{role}-{index}"
                identifiers.append(identifier)
                rows.append(
                    {
                        "utterance_id": identifier,
                        "speaker_id": speaker,
                        "chapter_id": f"chapter-{speaker}",
                        "split": split,
                        "wav_path": f"wav/{identifier}.wav",
                        "wav_sha256": hashlib.sha256(
                            f"wav:{identifier}".encode()
                        ).hexdigest(),
                        "source_sha256": hashlib.sha256(
                            f"source:{identifier}".encode()
                        ).hexdigest(),
                        "input_samples": 48000,
                        "padded_samples": 0,
                        "truncated_samples": 0,
                    }
                )
                values.append(vector.copy())
    manifest = {
        "format_version": 1,
        "dataset": "librispeech-dev-clean-pilot",
        "seed": 20261012,
        "rows": rows,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    embeddings_path = tmp_path / "teacher_embeddings.npz"
    np.savez(
        embeddings_path,
        ids=np.asarray(identifiers, dtype=str),
        embeddings=np.asarray(values, dtype=np.float32),
        manifest_sha256=np.asarray(manifest_sha256),
    )
    return manifest_path, embeddings_path, rows


def test_evaluate_embeddings_builds_all_same_split_trials(tmp_path) -> None:
    manifest_path, embeddings_path, rows = _write_embedding_fixture(tmp_path)
    result = evaluate_embeddings(manifest_path, embeddings_path, split="test")

    assert result["status"] == "NEW_LOCAL_LIBRISPEECH_PILOT"
    assert result["manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert result["embedding_shape"] == [len(rows), 192]
    assert result["enrollment_templates"] == 2
    assert result["query_utterances"] == 18
    assert result["score_trials"] == 36
    assert result["target_trials"] == 18
    assert result["non_target_trials"] == 18
    assert result["eer"] == pytest.approx(0.0)
    assert result["min_dcf_normalized"] == pytest.approx(0.0)
    assert len(result["scores"]) == 36
    json.dumps(result)


def test_evaluate_embeddings_rejects_manifest_hash_and_zero_vector(tmp_path) -> None:
    manifest_path, embeddings_path, _ = _write_embedding_fixture(tmp_path)
    with np.load(embeddings_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["manifest_sha256"] = np.asarray("0" * 64)
    bad_hash_path = tmp_path / "bad_hash.npz"
    np.savez(bad_hash_path, **arrays)
    with pytest.raises(ValueError, match="manifest_sha256"):
        evaluate_embeddings(manifest_path, bad_hash_path)

    arrays["manifest_sha256"] = np.asarray(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    arrays["embeddings"] = arrays["embeddings"].copy()
    arrays["embeddings"][0] = 0.0
    zero_path = tmp_path / "zero.npz"
    np.savez(zero_path, **arrays)
    with pytest.raises(ValueError, match="non-zero"):
        evaluate_embeddings(manifest_path, zero_path)


def test_evaluate_embeddings_rejects_speaker_leakage(tmp_path) -> None:
    manifest_path, embeddings_path, rows = _write_embedding_fixture(tmp_path)
    leaked = dict(rows[0])
    leaked["utterance_id"] = "leaked-train"
    leaked["split"] = "train"
    leaked["wav_path"] = "wav/leaked-train.wav"
    leaked["wav_sha256"] = "a" * 64
    leaked["source_sha256"] = "b" * 64
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rows"].append(leaked)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="speaker leakage"):
        evaluate_embeddings(manifest_path, embeddings_path)
