from __future__ import annotations

import numpy as np
import pytest

from edgeai.matching import (
    cosine_similarity,
    enroll,
    l2_distance,
    match_1_to_1,
    match_1_to_n,
    normalize_embedding,
)


def test_l2_cosine_and_enrollment_are_explicit() -> None:
    assert l2_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(np.sqrt(2.0))
    assert cosine_similarity([2.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    prototype = enroll(np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32))
    np.testing.assert_allclose(prototype, [np.sqrt(0.5), np.sqrt(0.5)])
    assert normalize_embedding(prototype).dtype == np.float32


@pytest.mark.parametrize(
    "bad_embedding",
    [
        [0.0, 0.0],
        [np.nan, 0.0],
        [np.inf, 1.0],
    ],
)
def test_zero_and_nonfinite_embeddings_are_rejected(bad_embedding) -> None:
    with pytest.raises(ValueError):
        normalize_embedding(bad_embedding)
    with pytest.raises(ValueError):
        cosine_similarity(bad_embedding, [1.0, 0.0])


def test_1_to_n_requires_threshold_and_margin_and_rejects_unknown() -> None:
    candidates = {
        "alice": np.array([1.0, 0.0], dtype=np.float32),
        "bob": np.array([0.0, 1.0], dtype=np.float32),
    }
    accepted = match_1_to_n(
        [1.0, 0.02], candidates, threshold=0.90, margin=0.10
    )
    assert accepted.accepted is True
    assert accepted.label == "alice"
    assert accepted.reason == "accepted"
    assert accepted.margin is not None and accepted.margin >= 0.10

    unknown = match_1_to_n(
        [1.0, 1.0], candidates, threshold=0.90, margin=0.10
    )
    assert unknown.accepted is False
    assert unknown.label is None
    assert unknown.reason == "below_threshold"

    ambiguous = match_1_to_n(
        [1.0, 0.02], candidates, threshold=0.0, margin=1.1
    )
    assert ambiguous.accepted is False
    assert ambiguous.reason == "insufficient_margin"


def test_1_to_1_and_l2_matching_use_caller_thresholds() -> None:
    verification = match_1_to_1(
        [0.0, 1.0], [0.0, 2.0], threshold=0.99, label="bob"
    )
    assert verification.accepted is True
    assert verification.label == "bob"

    l2_rejected = match_1_to_n(
        [1.0, 0.0], {"far": [0.0, 2.0]}, threshold=0.5, margin=0.0, metric="l2"
    )
    assert l2_rejected.accepted is False
    assert l2_rejected.reason == "above_distance_threshold"


def test_matching_does_not_supply_calibrated_threshold_defaults() -> None:
    with pytest.raises(TypeError):
        match_1_to_n([1.0, 0.0], {"a": [1.0, 0.0]})
    with pytest.raises(TypeError):
        match_1_to_1([1.0, 0.0], [1.0, 0.0])
