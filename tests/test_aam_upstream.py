"""Comparative tests against the installed SpeechBrain AAM arithmetic.

These tests are intentionally lazy: the project module is imported only after
the test process starts, and the independent PyTorch oracle runs in the
repository-local teacher environment because the main environment has no
PyTorch/SpeechBrain installation.  All inputs are deterministic synthetic
arrays; no project data or model artifacts are used.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TEACHER_PYTHON = REPO_ROOT / ".venv-teacher" / "bin" / "python"
EXPECTED_SPEECHBRAIN_VERSION = "1.1.1"
EXPECTED_TORCH_VERSION = "2.8.0"
EXPECTED_SOURCE_HASHES = {
    "losses.py": "cfff833f0827915bb56063f55aae760ee39fcc272d20980f4822805a9b585026",
    "ECAPA_TDNN.py": "5d1eec5fb66b8778962e7e97c923e423593bed2c82107197624be4874fd5a2d6",
}
MARGIN = 0.2
SCALE = 30.0
THRESHOLD = float(np.cos(np.pi - MARGIN))

COSINE_CASE = np.asarray(
    [
        [-1.0, -0.9999999, -0.2, 0.0, 0.2, 1.0],
        [THRESHOLD - 1e-6, THRESHOLD, THRESHOLD + 1e-6, -0.5, 0.5, 0.9999999],
        [1.2, -1.2, 0.01, -0.01, 0.8, -0.8],
    ],
    dtype=np.float32,
)
LABELS_CASE = np.asarray([0, 1, 4], dtype=np.int32)
EMBEDDINGS_CASE = np.asarray(
    [[3.0, 4.0, 0.0], [-2.0, 1.0, 2.0], [0.5, -1.5, 2.5]],
    dtype=np.float32,
)
KERNEL_CASE = np.asarray(
    [
        [1.0, 0.0, -1.0, 0.5],
        [0.0, 2.0, 1.0, -0.5],
        [1.0, 1.0, 0.0, 2.0],
    ],
    dtype=np.float32,
)


_ORACLE_SCRIPT = r"""
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import speechbrain
from speechbrain.lobes.models.ECAPA_TDNN import Classifier
from speechbrain.nnet.losses import AdditiveAngularMargin, LogSoftmaxWrapper

EXPECTED_SPEECHBRAIN_VERSION = "1.1.1"
EXPECTED_TORCH_VERSION = "2.8.0"
EXPECTED_SOURCE_HASHES = {
    "losses.py": "cfff833f0827915bb56063f55aae760ee39fcc272d20980f4822805a9b585026",
    "ECAPA_TDNN.py": "5d1eec5fb66b8778962e7e97c923e423593bed2c82107197624be4874fd5a2d6",
}


def source_hash(obj):
    path = Path(inspect.getsourcefile(obj)).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = EXPECTED_SOURCE_HASHES[path.name]
    if digest != expected:
        raise RuntimeError(f"unexpected {path.name} SHA256: {digest}")
    return {"path": str(path), "sha256": digest}


def main():
    if speechbrain.__version__ != EXPECTED_SPEECHBRAIN_VERSION:
        raise RuntimeError(f"unexpected SpeechBrain version: {speechbrain.__version__}")
    torch_version = torch.__version__.split("+")[0]
    if torch_version != EXPECTED_TORCH_VERSION:
        raise RuntimeError(f"unexpected PyTorch version: {torch.__version__}")

    payload = json.load(sys.stdin)
    cosine_values = np.asarray(payload["cosine"], dtype=np.float64)
    labels_values = np.asarray(payload["labels"], dtype=np.int64)
    embeddings_values = np.asarray(payload["embeddings"], dtype=np.float32)
    kernel_values = np.asarray(payload["kernel"], dtype=np.float32)
    margin = float(payload["margin"])
    scale = float(payload["scale"])
    easy_margin = bool(payload["easy_margin"])

    source_hashes = {
        "losses.py": source_hash(AdditiveAngularMargin),
        "ECAPA_TDNN.py": source_hash(Classifier),
    }

    cosine = torch.tensor(cosine_values, dtype=torch.float64)
    labels = torch.tensor(labels_values, dtype=torch.long)
    one_hot = F.one_hot(labels, num_classes=cosine.shape[1]).float()
    aam = AdditiveAngularMargin(
        margin=margin, scale=scale, easy_margin=easy_margin
    )
    aam_logits = aam(cosine, one_hot)

    wrapped = LogSoftmaxWrapper(aam)
    wrapped_loss = wrapped(cosine.unsqueeze(1), labels.unsqueeze(1))

    classifier = Classifier(
        input_size=embeddings_values.shape[1],
        lin_blocks=0,
        out_neurons=kernel_values.shape[1],
    )
    with torch.no_grad():
        classifier.weight.copy_(torch.tensor(kernel_values.T, dtype=torch.float32))
    embeddings = torch.tensor(embeddings_values, dtype=torch.float32)
    classifier_cosine = classifier(embeddings.unsqueeze(1)).squeeze(1)

    gradient_cosine = torch.tensor(
        cosine_values, dtype=torch.float32, requires_grad=True
    )
    gradient_labels = torch.tensor(labels_values, dtype=torch.long)
    gradient_loss = LogSoftmaxWrapper(
        AdditiveAngularMargin(
            margin=margin, scale=scale, easy_margin=easy_margin
        )
    )(
        gradient_cosine.unsqueeze(1), gradient_labels.unsqueeze(1)
    )
    gradient_loss.backward()

    gradient_embeddings = torch.tensor(
        embeddings_values, dtype=torch.float32, requires_grad=True
    )
    gradient_classifier = Classifier(
        input_size=embeddings_values.shape[1],
        lin_blocks=0,
        out_neurons=kernel_values.shape[1],
    )
    with torch.no_grad():
        gradient_classifier.weight.copy_(
            torch.tensor(kernel_values.T, dtype=torch.float32)
        )
    gradient_classifier_loss = (
        gradient_classifier(gradient_embeddings.unsqueeze(1)) ** 2
    ).mean()
    gradient_classifier_loss.backward()

    result = {
        "speechbrain_version": speechbrain.__version__,
        "torch_version": torch_version,
        "source_hashes": source_hashes,
        "aam_shape": list(aam_logits.shape),
        "aam_dtype": str(aam_logits.dtype),
        "aam_logits": aam_logits.detach().cpu().numpy().tolist(),
        "wrapper_loss": float(wrapped_loss.detach().cpu().item()),
        "classifier_shape": list(classifier_cosine.shape),
        "classifier_dtype": str(classifier_cosine.dtype),
        "classifier_cosine": classifier_cosine.detach().cpu().numpy().tolist(),
        "aam_gradient_finite": bool(
            gradient_cosine.grad is not None
            and torch.isfinite(gradient_cosine.grad).all()
        ),
        "classifier_gradient_finite": bool(
            gradient_embeddings.grad is not None
            and torch.isfinite(gradient_embeddings.grad).all()
        ),
    }
    print(json.dumps(result, sort_keys=True))


main()
"""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _run_official_oracle(
    *,
    cosine: np.ndarray,
    labels: np.ndarray,
    embeddings: np.ndarray = EMBEDDINGS_CASE,
    kernel: np.ndarray = KERNEL_CASE,
    margin: float = MARGIN,
    scale: float = SCALE,
    easy_margin: bool = False,
) -> dict[str, Any]:
    if not TEACHER_PYTHON.is_file():
        pytest.skip(f"official oracle interpreter is unavailable: {TEACHER_PYTHON}")
    payload = {
        "cosine": _jsonable(cosine),
        "labels": _jsonable(labels),
        "embeddings": _jsonable(embeddings),
        "kernel": _jsonable(kernel),
        "margin": margin,
        "scale": scale,
        "easy_margin": easy_margin,
    }
    completed = subprocess.run(
        [str(TEACHER_PYTHON), "-c", _ORACLE_SCRIPT],
        cwd=REPO_ROOT,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, (
        "official SpeechBrain oracle failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"official oracle did not emit JSON: {completed.stdout!r}\n"
            f"stderr:\n{completed.stderr}"
        ) from exc
    assert result["speechbrain_version"] == EXPECTED_SPEECHBRAIN_VERSION
    assert result["torch_version"] == EXPECTED_TORCH_VERSION
    assert {
        name: details["sha256"]
        for name, details in result["source_hashes"].items()
    } == EXPECTED_SOURCE_HASHES
    return result


def _local_api() -> tuple[Any, Callable[..., Any], Callable[..., Any]]:
    try:
        tf = importlib.import_module("tensorflow")
    except ModuleNotFoundError:
        pytest.skip("TensorFlow is unavailable in the local test environment")
    try:
        module = importlib.import_module("edgeai.comparative_training")
    except ModuleNotFoundError as exc:
        pytest.skip(f"comparative_training API is not available yet: {exc}")
    for name in ("normalized_classifier", "aam_logits"):
        if not hasattr(module, name):
            pytest.skip(f"comparative_training.{name} is not available yet")
    return tf, module.normalized_classifier, module.aam_logits


def _local_aam(
    tf: Any,
    aam_logits: Callable[..., Any],
    cosine: np.ndarray,
    labels: np.ndarray,
    *,
    margin: float = MARGIN,
    scale: float = SCALE,
    easy_margin: bool = False,
) -> np.ndarray:
    values = aam_logits(
        tf,
        tf.convert_to_tensor(cosine, dtype=tf.float32),
        tf.convert_to_tensor(labels, dtype=tf.int32),
        margin=margin,
        scale=scale,
        easy_margin=easy_margin,
    )
    return np.asarray(values.numpy(), dtype=np.float32)


def test_aam_interior_clamp_and_zero_margin_match_official() -> None:
    tf, _, aam_logits = _local_api()
    for easy_margin in (False, True):
        official = _run_official_oracle(
            cosine=COSINE_CASE,
            labels=LABELS_CASE,
            easy_margin=easy_margin,
        )
        local = _local_aam(
            tf,
            aam_logits,
            COSINE_CASE,
            LABELS_CASE,
            easy_margin=easy_margin,
        )
        np.testing.assert_allclose(
            local,
            np.asarray(official["aam_logits"], dtype=np.float32),
            atol=5e-6,
            rtol=1e-6,
        )
        assert local.shape == (3, 6)
        assert np.isfinite(local).all()

    official_m0 = _run_official_oracle(
        cosine=COSINE_CASE,
        labels=LABELS_CASE,
        margin=0.0,
        scale=1.0,
        easy_margin=False,
    )
    local_m0 = _local_aam(
        tf,
        aam_logits,
        COSINE_CASE,
        LABELS_CASE,
        margin=0.0,
        scale=1.0,
        easy_margin=False,
    )
    np.testing.assert_allclose(
        local_m0,
        np.asarray(official_m0["aam_logits"], dtype=np.float32),
        atol=5e-6,
        rtol=1e-6,
    )


def test_aam_strict_branch_equalities_match_official() -> None:
    tf, _, aam_logits = _local_api()
    branch_cosine = np.asarray(
        [
            [0.0, 0.2, -0.2, 0.0],
            [np.float32(THRESHOLD)] * 4,
            [np.float32(THRESHOLD + 1e-6)] * 4,
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 2], dtype=np.int32)
    for easy_margin in (False, True):
        official = _run_official_oracle(
            cosine=branch_cosine,
            labels=labels,
            easy_margin=easy_margin,
        )
        local = _local_aam(
            tf,
            aam_logits,
            branch_cosine,
            labels,
            easy_margin=easy_margin,
        )
        np.testing.assert_allclose(
            local,
            np.asarray(official["aam_logits"], dtype=np.float32),
            atol=5e-6,
            rtol=1e-6,
        )


def test_normalized_classifier_kernel_axes_match_official() -> None:
    tf, normalized_classifier, _ = _local_api()
    official = _run_official_oracle(
        cosine=COSINE_CASE,
        labels=LABELS_CASE,
    )
    local = normalized_classifier(
        tf,
        tf.convert_to_tensor(EMBEDDINGS_CASE, dtype=tf.float32),
        tf.convert_to_tensor(KERNEL_CASE, dtype=tf.float32),
    )
    local = np.asarray(local.numpy(), dtype=np.float32)
    np.testing.assert_allclose(
        local,
        np.asarray(official["classifier_cosine"], dtype=np.float32),
        atol=1e-6,
        rtol=1e-6,
    )
    assert local.shape == (3, 4)
    assert np.isfinite(local).all()
    assert official["classifier_shape"] == [3, 4]
    assert official["classifier_dtype"] == "torch.float32"


def test_mean_cross_entropy_matches_logsoftmax_wrapper() -> None:
    tf, _, aam_logits = _local_api()
    official = _run_official_oracle(
        cosine=COSINE_CASE,
        labels=LABELS_CASE,
    )
    local_logits = _local_aam(tf, aam_logits, COSINE_CASE, LABELS_CASE)
    local_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.convert_to_tensor(LABELS_CASE, dtype=tf.int32),
            logits=tf.convert_to_tensor(local_logits, dtype=tf.float32),
        )
    )
    np.testing.assert_allclose(
        float(local_loss.numpy()),
        float(official["wrapper_loss"]),
        atol=1e-5,
        rtol=1e-6,
    )


def test_local_and_official_gradients_are_finite() -> None:
    tf, normalized_classifier, aam_logits = _local_api()
    official = _run_official_oracle(
        cosine=COSINE_CASE,
        labels=LABELS_CASE,
    )
    assert official["aam_gradient_finite"] is True
    assert official["classifier_gradient_finite"] is True

    cosine = tf.Variable(COSINE_CASE, dtype=tf.float32)
    labels = tf.convert_to_tensor(LABELS_CASE, dtype=tf.int32)
    with tf.GradientTape() as tape:
        logits = aam_logits(
            tf,
            cosine,
            labels,
            margin=MARGIN,
            scale=SCALE,
            easy_margin=False,
        )
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        )
    gradient = tape.gradient(loss, cosine)
    assert gradient is not None
    assert bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy())

    embeddings = tf.Variable(EMBEDDINGS_CASE, dtype=tf.float32)
    with tf.GradientTape() as tape:
        classifier_cosine = normalized_classifier(
            tf,
            embeddings,
            tf.convert_to_tensor(KERNEL_CASE, dtype=tf.float32),
        )
        classifier_loss = tf.reduce_mean(tf.square(classifier_cosine))
    embedding_gradient = tape.gradient(classifier_loss, embeddings)
    assert embedding_gradient is not None
    assert bool(tf.reduce_all(tf.math.is_finite(embedding_gradient)).numpy())


@pytest.mark.parametrize(
    "case", ("zero_embedding", "nonfinite_embedding", "nonfinite_cosine")
)
def test_local_invalid_input_rejection_is_optional_local_design(case: str) -> None:
    """Accept local validation when exposed; do not attribute it to SpeechBrain."""

    tf, normalized_classifier, aam_logits = _local_api()
    if case == "zero_embedding":
        call = lambda: normalized_classifier(
            tf,
            tf.zeros((1, EMBEDDINGS_CASE.shape[1]), dtype=tf.float32),
            tf.convert_to_tensor(KERNEL_CASE, dtype=tf.float32),
        )
    elif case == "nonfinite_embedding":
        values = np.full((1, EMBEDDINGS_CASE.shape[1]), np.nan, dtype=np.float32)
        call = lambda: normalized_classifier(
            tf,
            tf.convert_to_tensor(values),
            tf.convert_to_tensor(KERNEL_CASE, dtype=tf.float32),
        )
    else:
        values = COSINE_CASE.copy()
        values[0, 0] = np.nan
        call = lambda: aam_logits(
            tf,
            tf.convert_to_tensor(values),
            tf.convert_to_tensor(LABELS_CASE, dtype=tf.int32),
            margin=MARGIN,
            scale=SCALE,
            easy_margin=False,
        )

    try:
        result = call()
        if hasattr(result, "numpy"):
            result.numpy()
    except (FloatingPointError, TypeError, ValueError, tf.errors.InvalidArgumentError):
        return
    pytest.skip(
        "local invalid-input rejection is not exposed; upstream arithmetic "
        "also performs no such validation"
    )
