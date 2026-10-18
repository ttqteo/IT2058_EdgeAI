"""Small, explicit embedding matching primitives for the host prototype."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping

import numpy as np

Metric = Literal["cosine", "l2"]


def _as_vector(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real-valued numbers")
    try:
        vector = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def normalize_embedding(embedding: object) -> np.ndarray:
    """Return an L2-unit float32 embedding, rejecting zero/nonfinite input."""

    vector = _as_vector(embedding, "embedding")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("embedding must have a finite, non-zero L2 norm")
    return (vector / norm).astype(np.float32)


def _same_shape(left: object, right: object) -> tuple[np.ndarray, np.ndarray]:
    left_vector = _as_vector(left, "left embedding")
    right_vector = _as_vector(right, "right embedding")
    if left_vector.shape != right_vector.shape:
        raise ValueError(
            "embedding dimensions must match; got "
            f"{left_vector.size} and {right_vector.size}"
        )
    return left_vector, right_vector


def l2_distance(left: object, right: object) -> float:
    """Return Euclidean distance between two finite, same-shaped embeddings."""

    left_vector, right_vector = _same_shape(left, right)
    distance = float(np.linalg.norm(left_vector - right_vector))
    if not np.isfinite(distance):
        raise ValueError("L2 distance is not finite")
    return distance


def cosine_similarity(left: object, right: object) -> float:
    """Return cosine similarity, rejecting either zero embedding."""

    left_vector, right_vector = _same_shape(left, right)
    left_unit = normalize_embedding(left_vector).astype(np.float64)
    right_unit = normalize_embedding(right_vector).astype(np.float64)
    similarity = float(np.dot(left_unit, right_unit))
    if not np.isfinite(similarity):
        raise ValueError("cosine similarity is not finite")
    return similarity


def enroll(embeddings: object) -> np.ndarray:
    """Enroll a prototype as the normalized mean of unit-normalized samples."""

    raw = np.asarray(embeddings)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty two-dimensional array")
    unit_embeddings = np.stack(
        [normalize_embedding(row) for row in raw], axis=0
    ).astype(np.float64)
    mean_embedding = np.mean(unit_embeddings, axis=0)
    return normalize_embedding(mean_embedding)


@dataclass(frozen=True)
class MatchResult:
    """Decision and scores from a caller-thresholded match."""

    accepted: bool
    label: str | None
    score: float | None
    second_score: float | None
    margin: float | None
    threshold: float
    required_margin: float | None
    metric: Metric
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_metric(metric: str) -> Metric:
    if metric not in {"cosine", "l2"}:
        raise ValueError("metric must be 'cosine' or 'l2'")
    return metric  # type: ignore[return-value]


def _validate_threshold(value: float, name: str) -> float:
    threshold = float(value)
    if not np.isfinite(threshold):
        raise ValueError(f"{name} must be finite")
    return threshold


def _validate_margin(value: float) -> float:
    margin = float(value)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("margin must be finite and non-negative")
    return margin


def _score(query: object, reference: object, metric: Metric) -> float:
    if metric == "cosine":
        return cosine_similarity(query, reference)
    return l2_distance(query, reference)


def match_1_to_n(
    query: object,
    candidates: Mapping[str, object],
    *,
    threshold: float,
    margin: float,
    metric: Metric = "cosine",
) -> MatchResult:
    """Run top-1 matching with caller-supplied threshold and margin.

    For cosine, larger scores are better and acceptance requires
    ``top1 >= threshold`` and ``top1 - top2 >= margin``.  For L2, smaller
    distances are better and acceptance requires ``top1 <= threshold`` and
    ``top2 - top1 >= margin``.  A one-candidate set has no runner-up, so its
    margin condition is considered satisfied; the threshold still applies.
    """

    selected_metric = _validate_metric(metric)
    threshold_value = _validate_threshold(threshold, "threshold")
    required_margin = _validate_margin(margin)
    if not candidates:
        raise ValueError("at least one candidate is required")

    scored = [
        (str(label), _score(query, reference, selected_metric))
        for label, reference in candidates.items()
    ]
    reverse = selected_metric == "cosine"
    scored.sort(key=lambda item: item[1], reverse=reverse)
    label, top_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else None
    score_margin = (
        None
        if second_score is None
        else (top_score - second_score if reverse else second_score - top_score)
    )
    threshold_pass = (
        top_score >= threshold_value
        if reverse
        else top_score <= threshold_value
    )
    margin_pass = score_margin is None or score_margin >= required_margin
    accepted = threshold_pass and margin_pass
    if not threshold_pass:
        reason = "below_threshold" if reverse else "above_distance_threshold"
    elif not margin_pass:
        reason = "insufficient_margin"
    else:
        reason = "accepted"

    return MatchResult(
        accepted=accepted,
        label=label if accepted else None,
        score=float(top_score),
        second_score=None if second_score is None else float(second_score),
        margin=None if score_margin is None else float(score_margin),
        threshold=threshold_value,
        required_margin=required_margin,
        metric=selected_metric,
        reason=reason,
    )


def match_1_to_1(
    query: object,
    reference: object,
    *,
    threshold: float,
    metric: Metric = "cosine",
    label: str | None = None,
) -> MatchResult:
    """Run caller-thresholded 1:1 verification without a calibrated default."""

    selected_metric = _validate_metric(metric)
    threshold_value = _validate_threshold(threshold, "threshold")
    score = _score(query, reference, selected_metric)
    accepted = (
        score >= threshold_value
        if selected_metric == "cosine"
        else score <= threshold_value
    )
    if accepted:
        reason = "accepted"
    else:
        reason = (
            "below_threshold"
            if selected_metric == "cosine"
            else "above_distance_threshold"
        )
    return MatchResult(
        accepted=accepted,
        label=label if accepted else None,
        score=float(score),
        second_score=None,
        margin=None,
        threshold=threshold_value,
        required_margin=None,
        metric=selected_metric,
        reason=reason,
    )


__all__ = [
    "MatchResult",
    "cosine_similarity",
    "enroll",
    "l2_distance",
    "match_1_to_1",
    "match_1_to_n",
    "normalize_embedding",
]
