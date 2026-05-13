from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch.nn.functional import log_softmax

from iat.models import LoadedModel


def _kl_div(log_p: torch.Tensor, log_q: torch.Tensor) -> float:
    """KL(P || Q) = sum_v p(v) (log p(v) - log q(v)). Inputs son log-probs (V,)."""
    p = log_p.exp()
    return float((p * (log_p - log_q)).sum().item())


@torch.no_grad()
def compute_m1_cf(
    lm: LoadedModel,
    token_ids: torch.Tensor,
    positions: Sequence[int],
    k: int = 5,
    dtype_kl: torch.dtype = torch.float32,
) -> list[dict[str, float | int]]:
    """
    M1_cf — Counterfactual block KL contra argmax del modelo.

    Definicion:
        v_top(i) = argmax_v p(v | t_0..t_{i-1})
        M1_cf(i) = sum_{j=1..k} KL( p(. | t_0..t_{i+j-1})
                                   || p(. | t_0..t_{i-1}, v_top, t_{i+1..i+j-1}) )

    Implementacion:
        - Forward A: la secuencia completa token_ids.
        - Forward B: la misma secuencia con t_i reemplazado por v_top en la
          posicion i. Mismas longitudes que A, mismo causal mask, mismas
          posiciones para todas las predicciones; las dos distribuciones
          comparadas en cada KL viven sobre la misma posicion (i+j) y solo
          difieren en si el modelo vio t_i o v_top en la posicion i.

    Precision asimetrica: el forward del modelo usa lm.dtype (bfloat16 en
    MPS); el log_softmax y el KL se hacen en dtype_kl (default float32)
    para evitar el ruido bf16 observado en exp_004 al exponenciar logits.

    Tambien devuelve delta_cloze_cf por target:
        delta_cloze_cf(i) = | log p(t_{i+k} | C, t_i) - log p(t_{i+k} | C, v_top) |
    util como rama T1 (probe cloze a k pasos vista) sin el bug de skip-shift
    de exp_004.

    Args:
        lm: modelo cargado.
        token_ids: tensor (1, L) con ids.
        positions: target_pos por evaluar; cada i debe cumplir
            1 <= i <= L - 1 - k.
        k: tamano del bloque futuro. Default 5.
        dtype_kl: precision usada para log_softmax y el sum de KL.

    Returns:
        Lista de dicts en el orden de positions; keys:
            target_pos, v_top, surprisal_bits, m1_cf_kl_nats, delta_cloze_cf.
    """
    if token_ids.dim() != 2 or token_ids.shape[0] != 1:
        raise ValueError(f"token_ids debe tener shape (1, L), recibido {tuple(token_ids.shape)}")
    seq_len = token_ids.shape[1]
    if k < 1:
        raise ValueError(f"k debe ser >= 1, recibido {k}")

    token_ids = token_ids.to(lm.device)

    out_a = lm.model(input_ids=token_ids)
    logits_a = out_a.logits[0]  # (L, V) en dtype del modelo

    results: list[dict[str, float | int]] = []
    for i in positions:
        if not 1 <= i <= seq_len - 1 - k:
            raise ValueError(
                f"target {i} fuera de rango; se requiere 1 <= i <= L - 1 - k "
                f"con L={seq_len}, k={k} (rango valido: [1, {seq_len - 1 - k}])"
            )

        v_top = int(logits_a[i - 1].argmax().item())

        ids_b = token_ids.clone()
        ids_b[0, i] = v_top
        out_b = lm.model(input_ids=ids_b)
        logits_b = out_b.logits[0]

        m1_cf_total = 0.0
        for j in range(1, k + 1):
            pos = i + j - 1
            log_p_a = log_softmax(logits_a[pos].to(dtype_kl), dim=-1)
            log_p_b = log_softmax(logits_b[pos].to(dtype_kl), dim=-1)
            m1_cf_total += _kl_div(log_p_a, log_p_b)

        target_token = int(token_ids[0, i].item())
        log_p_a_at_im1 = log_softmax(logits_a[i - 1].to(dtype_kl), dim=-1)
        nll_nats = -float(log_p_a_at_im1[target_token].item())
        surprisal_bits = nll_nats / math.log(2.0)

        t_target = int(token_ids[0, i + k].item())
        log_p_a_at_k = log_softmax(logits_a[i + k - 1].to(dtype_kl), dim=-1)
        log_p_b_at_k = log_softmax(logits_b[i + k - 1].to(dtype_kl), dim=-1)
        lpa = float(log_p_a_at_k[t_target].item())
        lpb = float(log_p_b_at_k[t_target].item())
        delta_cloze_cf = abs(lpa - lpb)

        results.append(
            {
                "target_pos": int(i),
                "v_top": v_top,
                "surprisal_bits": surprisal_bits,
                "m1_cf_kl_nats": m1_cf_total,
                "delta_cloze_cf": delta_cloze_cf,
            }
        )

    return results


@torch.no_grad()
def compute_m1_marginal(
    lm: LoadedModel,
    token_ids: torch.Tensor,
    positions: Sequence[int],
    top_k: int = 32,
    dtype_kl: torch.dtype = torch.float32,
) -> list[dict[str, float | int]]:
    """
    M1_marginal — KL contra marginal aproximada con top-K.

    Definicion (k=1, un solo token siguiente):
        p_marginal(t_{i+1} | C) ~= sum_{v en topK} (p(v|C) / Z) * p(t_{i+1} | C, v)
        donde Z = sum_{v en topK} p(v|C) (renormalizacion a 1 sobre topK)
              C = t_0..t_{i-1}
        M1_marginal(i) = KL( p(. | t_0..t_i) || p_marginal(. | t_0..t_{i-1}) )

    Implementacion:
        - Forward A: secuencia completa.
        - Para cada target i, identifica top_k candidatos por p(v|C) leido
          desde logits_A[i-1].
        - Forward batched de tamano top_k sobre las secuencias
          [t_0..t_{i-1}, v_j] (longitud i+1). Cada fila comparte el prefix
          C; difieren solo en la sustitucion en la posicion i.
        - logits del batch en la posicion i = p(. | C, v_j) para cada j.
        - p_marginal = sum_j (p(v_j|C)/Z) * p(. | C, v_j).

    Precision asimetrica: forward en dtype del modelo, log_softmax y KL en
    dtype_kl (default float32).

    Args:
        lm: modelo cargado.
        token_ids: tensor (1, L).
        positions: target_pos por evaluar; cada i debe cumplir
            1 <= i <= L - 2 (para que t_{i+1} exista).
        top_k: numero de candidatos para la marginal aproximada.
        dtype_kl: precision para log_softmax y KL.

    Returns:
        Lista de dicts en el orden de positions; keys:
            target_pos, surprisal_bits, m1_marginal_kl_nats, t_i_in_topk,
            top_k_mass_pre_renorm.
    """
    if token_ids.dim() != 2 or token_ids.shape[0] != 1:
        raise ValueError(f"token_ids debe tener shape (1, L), recibido {tuple(token_ids.shape)}")
    seq_len = token_ids.shape[1]
    if top_k < 1:
        raise ValueError(f"top_k debe ser >= 1, recibido {top_k}")
    vocab = int(lm.model.config.vocab_size)
    if top_k > vocab:
        raise ValueError(f"top_k={top_k} excede vocab size {vocab}")

    token_ids = token_ids.to(lm.device)

    out_a = lm.model(input_ids=token_ids)
    logits_a = out_a.logits[0]

    results: list[dict[str, float | int]] = []
    for i in positions:
        if not 1 <= i <= seq_len - 2:
            raise ValueError(
                f"target {i} fuera de rango; se requiere 1 <= i <= L - 2 "
                f"con L={seq_len} (rango valido: [1, {seq_len - 2}])"
            )

        log_p_v = log_softmax(logits_a[i - 1].to(dtype_kl), dim=-1)  # (V,)
        top_vals_log, top_indices = log_p_v.topk(top_k)
        top_vals = top_vals_log.exp()  # (top_k,)
        mass_pre = float(top_vals.sum().item())
        weights = top_vals / top_vals.sum()  # (top_k,) suma a 1 estrictamente

        prefix = token_ids[:, :i]  # (1, i)
        prefix_repeat = prefix.expand(top_k, -1)  # (top_k, i)
        v_col = top_indices.view(-1, 1)  # (top_k, 1)
        batch_input = torch.cat([prefix_repeat, v_col], dim=1)  # (top_k, i+1)

        out_batch = lm.model(input_ids=batch_input)
        logits_at_i = out_batch.logits[:, i, :].to(dtype_kl)  # (top_k, V) en posicion i
        log_p_after_v = log_softmax(logits_at_i, dim=-1)  # (top_k, V)
        p_after_v = log_p_after_v.exp()
        p_marginal = (weights.unsqueeze(1) * p_after_v).sum(dim=0)  # (V,)
        log_p_marginal = p_marginal.clamp_min(1e-30).log()

        log_p_a_after_ti = log_softmax(logits_a[i].to(dtype_kl), dim=-1)
        p_a_after_ti = log_p_a_after_ti.exp()
        m1_marg = float((p_a_after_ti * (log_p_a_after_ti - log_p_marginal)).sum().item())

        target_token = int(token_ids[0, i].item())
        nll_nats = -float(log_p_v[target_token].item())
        surprisal_bits = nll_nats / math.log(2.0)
        t_i_in_topk = bool((top_indices == target_token).any().item())

        results.append(
            {
                "target_pos": int(i),
                "surprisal_bits": surprisal_bits,
                "m1_marginal_kl_nats": m1_marg,
                "t_i_in_topk": int(t_i_in_topk),
                "top_k_mass_pre_renorm": mass_pre,
            }
        )

    return results
