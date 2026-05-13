from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch.nn.functional import log_softmax

from iat.models import LoadedModel


@torch.no_grad()
def m1_and_cloze_block(
    lm: LoadedModel,
    token_ids: torch.Tensor,
    targets: Sequence[int],
    k: int = 5,
) -> list[dict[str, float | int]]:
    """
    Bayesian surprise M1 sobre bloque futuro y delta cloze por target token,
    siguiendo `docs/design.md` §3 (M1) y §4 (T1).

    Por cada target_pos i (0-indexed sobre la secuencia):
      - Forward A en la secuencia completa (compartido entre targets dentro
        de la misma llamada).
      - Forward B en la secuencia con t_i omitido:
        [t_0..t_{i-1}, t_{i+1}..t_{L-1}], shape (1, L-1).
      - p_after_j  = softmax(logits_A[i + j - 1])  para j = 1..k
        ≡ p(t_{i+j} | t_0..t_{i+j-1})  (condiciona en t_i y en la
        continuación real teacher-forced).
      - p_before_j = softmax(logits_B[i + j - 2])  para j = 1..k
        ≡ p(t_{i+j} | t_0..t_{i-1}, t_{i+1}, ..., t_{i+j-1})  (NO
        condiciona en t_i; usa la continuación real para teacher-forcing).
      - M1(i) = Σ_{j=1..k} KL(p_after_j || p_before_j), en nats.
      - delta_cloze(i) = | log p_after_k(t_{i+k}) - log p_before_k(t_{i+k}) |.

    Args:
        lm: modelo cargado (output_hidden_states irrelevante aquí).
        token_ids: tensor (1, L) con ids.
        targets: posiciones a evaluar; cada i debe cumplir
            `1 <= i <= L - 1 - k` para que t_{i+k} exista y t_{i-1} no sea
            negativo. ValueError si alguna está fuera.
        k: tamaño del bloque futuro. Default 5.

    Returns:
        Lista de dicts en el mismo orden que `targets`, con keys:
        `target_pos`, `surprisal_bits`, `m1_kl_block_nats`, `delta_cloze`.
        `surprisal_bits` se calcula también desde forward A (Shannon en bits)
        para conveniencia y alineación inmediata; coincidirá con
        `shannon_surprisal` token-a-token.

    Coste: 1 forward A + len(targets) forwards B. Para 100 docs x ~200
    targets/doc en Pythia 1.4B aprox 20 000 forwards. Sin reuso de KV-cache.
    """
    if token_ids.dim() != 2 or token_ids.shape[0] != 1:
        raise ValueError(f"token_ids debe tener shape (1, L), recibido {tuple(token_ids.shape)}")
    seq_len = token_ids.shape[1]
    if k < 1:
        raise ValueError(f"k debe ser >= 1, recibido {k}")

    token_ids = token_ids.to(lm.device)

    out_a = lm.model(input_ids=token_ids)
    logits_a = out_a.logits[0].float()  # (L, V)
    log_p_a_full = log_softmax(logits_a, dim=-1)  # (L, V)

    results: list[dict[str, float | int]] = []
    for i in targets:
        if not 1 <= i <= seq_len - 1 - k:
            raise ValueError(
                f"target {i} fuera de rango; se requiere 1 <= i <= L - 1 - k "
                f"con L={seq_len}, k={k} (rango valido: [1, {seq_len - 1 - k}])"
            )

        ids_b = torch.cat([token_ids[:, :i], token_ids[:, i + 1 :]], dim=1)
        out_b = lm.model(input_ids=ids_b)
        logits_b = out_b.logits[0].float()  # (L-1, V)
        log_p_b_full = log_softmax(logits_b, dim=-1)

        target_token = int(token_ids[0, i].item())
        nll_nats = -float(log_p_a_full[i - 1, target_token].item())
        surprisal_bits = nll_nats / math.log(2.0)

        m1_total = 0.0
        for j in range(1, k + 1):
            log_p_a = log_p_a_full[i + j - 1]
            log_p_b = log_p_b_full[i + j - 2]
            p_a = log_p_a.exp()
            kl = float((p_a * (log_p_a - log_p_b)).sum().item())
            m1_total += kl

        t_target = int(token_ids[0, i + k].item())
        idx_a_k = i + k - 1
        idx_b_k = i + k - 2
        log_p_a_target = float(log_p_a_full[idx_a_k, t_target].item())
        log_p_b_target = float(log_p_b_full[idx_b_k, t_target].item())
        delta_cloze = abs(log_p_a_target - log_p_b_target)

        results.append(
            {
                "target_pos": int(i),
                "surprisal_bits": surprisal_bits,
                "m1_kl_block_nats": m1_total,
                "delta_cloze": delta_cloze,
            }
        )

    return results
