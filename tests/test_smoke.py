from __future__ import annotations

import pytest
import torch

from iat.metrics.activation import (
    activation_drift_last_layer,
    activation_drift_multilayer,
)
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


@pytest.mark.parametrize("layer_idx", [0, 1, -2, -1])
def test_activation_drift_layer_idx_shape_and_range(layer_idx: int) -> None:
    lm: LoadedModel = load_model("hf-internal-testing/tiny-random-gpt2", dtype=torch.float32)
    ids = torch.tensor([[5, 10, 15, 20, 25]])
    out = activation_drift_last_layer(lm, ids, layer_idx=layer_idx)
    assert out.shape == (4,)
    assert (out >= 0).all()
    assert torch.isfinite(out).all()


def test_activation_drift_layer_idx_default_is_last() -> None:
    lm: LoadedModel = load_model("hf-internal-testing/tiny-random-gpt2", dtype=torch.float32)
    ids = torch.tensor([[5, 10, 15, 20, 25]])
    default = activation_drift_last_layer(lm, ids)
    explicit = activation_drift_last_layer(lm, ids, layer_idx=-1)
    assert torch.allclose(default, explicit)


def test_activation_drift_layer_idx_out_of_range() -> None:
    lm: LoadedModel = load_model("hf-internal-testing/tiny-random-gpt2", dtype=torch.float32)
    ids = torch.tensor([[5, 10, 15, 20, 25]])
    with pytest.raises(IndexError):
        activation_drift_last_layer(lm, ids, layer_idx=9999)


def test_activation_drift_multilayer_matches_single() -> None:
    lm: LoadedModel = load_model("hf-internal-testing/tiny-random-gpt2", dtype=torch.float32)
    ids = torch.tensor([[5, 10, 15, 20, 25]])
    indices = [0, 1, -1]
    multi = activation_drift_multilayer(lm, ids, indices)
    assert set(multi.keys()) == set(indices)
    for li in indices:
        single = activation_drift_last_layer(lm, ids, layer_idx=li)
        assert torch.allclose(multi[li], single)


def test_activation_drift_multilayer_rejects_empty() -> None:
    lm: LoadedModel = load_model("hf-internal-testing/tiny-random-gpt2", dtype=torch.float32)
    ids = torch.tensor([[5, 10, 15, 20, 25]])
    with pytest.raises(ValueError, match="layer_indices"):
        activation_drift_multilayer(lm, ids, [])
