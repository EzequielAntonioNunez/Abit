from __future__ import annotations

import torch

from iat.metrics.activation import activation_drift_last_layer
from iat.metrics.shannon import shannon_surprisal
from iat.models import LoadedModel, get_device, load_model


def test_get_device_returns_torch_device() -> None:
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cuda", "mps", "cpu"}


def test_shannon_surprisal_shape_and_range() -> None:
    lm: LoadedModel = load_model("hf-internal-testing/tiny-random-gpt2", dtype=torch.float32)
    ids = torch.tensor([[5, 10, 15, 20, 25]])
    out = shannon_surprisal(lm, ids)
    assert out.shape == (4,)
    assert (out >= 0).all()
    assert torch.isfinite(out).all()


def test_shannon_surprisal_rejects_bad_shape() -> None:
    lm: LoadedModel = load_model("hf-internal-testing/tiny-random-gpt2", dtype=torch.float32)
    bad = torch.tensor([5, 10, 15])
    try:
        shannon_surprisal(lm, bad)
    except ValueError:
        return
    raise AssertionError("se esperaba ValueError por shape inválido")


def test_activation_drift_last_layer_shape_and_range() -> None:
    lm: LoadedModel = load_model("hf-internal-testing/tiny-random-gpt2", dtype=torch.float32)
    ids = torch.tensor([[5, 10, 15, 20, 25]])
    out = activation_drift_last_layer(lm, ids)
    assert out.shape == (4,)
    assert (out >= 0).all()
    assert torch.isfinite(out).all()


def test_activation_drift_last_layer_rejects_bad_shape() -> None:
    lm: LoadedModel = load_model("hf-internal-testing/tiny-random-gpt2", dtype=torch.float32)
    bad = torch.tensor([5, 10, 15])
    try:
        activation_drift_last_layer(lm, bad)
    except ValueError:
        return
    raise AssertionError("se esperaba ValueError por shape inválido")


def test_metrics_aligned_token_count() -> None:
    lm: LoadedModel = load_model("hf-internal-testing/tiny-random-gpt2", dtype=torch.float32)
    ids = torch.tensor([[5, 10, 15, 20, 25, 30]])
    s = shannon_surprisal(lm, ids)
    d = activation_drift_last_layer(lm, ids)
    assert s.shape == d.shape
