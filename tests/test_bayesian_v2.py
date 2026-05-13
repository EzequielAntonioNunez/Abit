from __future__ import annotations

import math

import pytest
import torch

from iat.metrics.bayesian_v2 import compute_m1_cf, compute_m1_marginal
from iat.models import LoadedModel, load_model

MODEL_NAME = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def lm() -> LoadedModel:
    return load_model(MODEL_NAME, dtype=torch.float32)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_compute_m1_cf_shapes_and_keys(lm: LoadedModel, k: int) -> None:
    ids = torch.tensor([[5, 10, 15, 20, 25, 30, 35, 40]])
    out = compute_m1_cf(lm, ids, positions=[2, 3], k=k)
    assert len(out) == 2
    expected_keys = {
        "target_pos",
        "v_top",
        "surprisal_bits",
        "m1_cf_kl_nats",
        "delta_cloze_cf",
    }
    for row in out:
        assert set(row.keys()) == expected_keys
        assert math.isfinite(float(row["surprisal_bits"]))
        assert math.isfinite(float(row["m1_cf_kl_nats"]))
        assert math.isfinite(float(row["delta_cloze_cf"]))


@pytest.mark.parametrize("top_k", [1, 2, 4, 8])
def test_compute_m1_marginal_shapes_and_keys(lm: LoadedModel, top_k: int) -> None:
    ids = torch.tensor([[5, 10, 15, 20, 25, 30, 35, 40]])
    out = compute_m1_marginal(lm, ids, positions=[2, 3, 4], top_k=top_k)
    assert len(out) == 3
    expected_keys = {
        "target_pos",
        "surprisal_bits",
        "m1_marginal_kl_nats",
        "t_i_in_topk",
        "top_k_mass_pre_renorm",
    }
    for row in out:
        assert set(row.keys()) == expected_keys
        assert math.isfinite(float(row["m1_marginal_kl_nats"]))
        assert 0.0 < float(row["top_k_mass_pre_renorm"]) <= 1.0


@pytest.mark.parametrize("k", [1, 3, 5])
def test_m1_cf_non_negative(lm: LoadedModel, k: int) -> None:
    ids = torch.tensor([[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]])
    out = compute_m1_cf(lm, ids, positions=[2, 3, 4], k=k)
    for row in out:
        assert float(row["m1_cf_kl_nats"]) >= -1e-5
        assert float(row["delta_cloze_cf"]) >= 0.0


@pytest.mark.parametrize("top_k", [1, 4, 16])
def test_m1_marginal_non_negative(lm: LoadedModel, top_k: int) -> None:
    ids = torch.tensor([[5, 10, 15, 20, 25, 30, 35, 40, 45]])
    out = compute_m1_marginal(lm, ids, positions=[2, 3, 4, 5], top_k=top_k)
    for row in out:
        assert float(row["m1_marginal_kl_nats"]) >= -1e-5


def test_m1_marginal_topk_1_equals_m1_cf_k1(lm: LoadedModel) -> None:
    """
    Coherencia interna: M1_marginal con top_k=1 colapsa la marginal a
    p(. | C, v_top), que es exactamente la rama B de M1_cf con k=1.
    Las dos KL deben coincidir hasta ruido numerico.
    """
    ids = torch.tensor([[5, 10, 15, 20, 25, 30, 35, 40, 45]])
    positions = [2, 3, 4, 5]
    out_cf = compute_m1_cf(lm, ids, positions=positions, k=1)
    out_marg = compute_m1_marginal(lm, ids, positions=positions, top_k=1)
    for row_cf, row_marg in zip(out_cf, out_marg, strict=True):
        assert row_cf["target_pos"] == row_marg["target_pos"]
        kl_cf = float(row_cf["m1_cf_kl_nats"])
        kl_marg = float(row_marg["m1_marginal_kl_nats"])
        assert math.isclose(kl_cf, kl_marg, rel_tol=1e-3, abs_tol=1e-4), (
            f"top_k=1 deberia coincidir con k=1: cf={kl_cf}, marg={kl_marg}"
        )


def test_m1_cf_zero_when_ti_is_vtop(lm: LoadedModel) -> None:
    """
    Sanity: si forzamos t_i = v_top (la prediccion del modelo en posicion
    i), forward A y forward B son identicos por construccion, y todos los
    sumandos de M1_cf deben ser 0.
    """
    ids = torch.tensor([[5, 10, 15, 20, 25, 30, 35, 40]])
    with torch.no_grad():
        logits = lm.model(input_ids=ids.to(lm.device)).logits[0]
    i = 4
    v_top = int(logits[i - 1].argmax().item())
    ids_modified = ids.clone()
    ids_modified[0, i] = v_top
    out = compute_m1_cf(lm, ids_modified, positions=[i], k=2)
    assert math.isclose(float(out[0]["m1_cf_kl_nats"]), 0.0, abs_tol=1e-5)
    assert math.isclose(float(out[0]["delta_cloze_cf"]), 0.0, abs_tol=1e-5)


def test_m1_marginal_zero_when_ti_dominates(lm: LoadedModel) -> None:
    """
    Sanity: si t_i = v_top y top_k=1, la marginal es p(. | C, v_top) y la
    rama de t_i es p(. | C, t_i) = misma distribucion. KL ~ 0.
    """
    ids = torch.tensor([[5, 10, 15, 20, 25, 30, 35, 40]])
    with torch.no_grad():
        logits = lm.model(input_ids=ids.to(lm.device)).logits[0]
    i = 4
    v_top = int(logits[i - 1].argmax().item())
    ids_modified = ids.clone()
    ids_modified[0, i] = v_top
    out = compute_m1_marginal(lm, ids_modified, positions=[i], top_k=1)
    assert math.isclose(float(out[0]["m1_marginal_kl_nats"]), 0.0, abs_tol=1e-5)


def test_m1_cf_rejects_bad_target_range(lm: LoadedModel) -> None:
    ids = torch.tensor([[5, 10, 15, 20, 25, 30]])
    with pytest.raises(ValueError, match="target"):
        compute_m1_cf(lm, ids, positions=[0], k=2)
    with pytest.raises(ValueError, match="target"):
        compute_m1_cf(lm, ids, positions=[4], k=2)


def test_m1_marginal_rejects_bad_target_range(lm: LoadedModel) -> None:
    ids = torch.tensor([[5, 10, 15, 20, 25, 30]])
    with pytest.raises(ValueError, match="target"):
        compute_m1_marginal(lm, ids, positions=[0], top_k=2)
    with pytest.raises(ValueError, match="target"):
        compute_m1_marginal(lm, ids, positions=[5], top_k=2)


def test_m1_marginal_rejects_topk_too_large(lm: LoadedModel) -> None:
    ids = torch.tensor([[5, 10, 15, 20, 25, 30]])
    vocab = int(lm.model.config.vocab_size)
    with pytest.raises(ValueError, match="vocab"):
        compute_m1_marginal(lm, ids, positions=[2], top_k=vocab + 100)


def test_m1_cf_rejects_bad_shape(lm: LoadedModel) -> None:
    bad = torch.tensor([5, 10, 15])
    with pytest.raises(ValueError, match="shape"):
        compute_m1_cf(lm, bad, positions=[1], k=1)


def test_m1_marginal_rejects_bad_shape(lm: LoadedModel) -> None:
    bad = torch.tensor([5, 10, 15])
    with pytest.raises(ValueError, match="shape"):
        compute_m1_marginal(lm, bad, positions=[1], top_k=2)
