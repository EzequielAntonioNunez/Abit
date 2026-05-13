from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from iat.models import LoadedModel


def _drift_from_hidden(hidden: torch.Tensor) -> torch.Tensor:
    """
    Calcula ||h_i - h_{i-1}||_2 / sqrt(d) por posición.

    Args:
        hidden: tensor (1, L, d) o (L, d) con un único batch.

    Returns:
        tensor (L-1,) en float32.
    """
    if hidden.dim() == 3:
        hidden = hidden[0]
    h = hidden.float()  # (L, d)
    diff = h[1:] - h[:-1]  # (L-1, d)
    d = h.shape[-1]
    drift: torch.Tensor = diff.norm(dim=-1) / math.sqrt(d)
    return drift


@torch.no_grad()
def activation_drift_last_layer(
    lm: LoadedModel,
    token_ids: torch.Tensor,
    layer_idx: int = -1,
) -> torch.Tensor:
    """
    M2_l (norma L2 normalizada del cambio por posición) en la capa indicada.

    Por causalidad del masking en LMs autoregresivos, ambos estados se obtienen
    en un solo forward pass: `hidden_states[layer_idx]` tiene shape (1, L, d) y
    la fila i corresponde al estado tras consumir tokens t_1..t_i.

    `layer_idx = -1` (default) reproduce el comportamiento original: deriva en
    la última capa, post final layer-norm. Índices admitidos: cualquier valor
    válido para indexar `outputs.hidden_states`, que tiene longitud
    `n_layers + 1` (índice 0 = embeddings de entrada, índice -1 = última capa).

    Args:
        lm: modelo cargado; debe haberse cargado con output_hidden_states=True.
        token_ids: tensor (1, L) con ids de tokens.
        layer_idx: índice de capa en `outputs.hidden_states`. Default -1.

    Returns:
        tensor (L-1,) en float32. La posición i corresponde al token original
        i+1 (alineado con `shannon_surprisal`).
    """
    if token_ids.dim() != 2 or token_ids.shape[0] != 1:
        raise ValueError(f"token_ids debe tener shape (1, L), recibido {tuple(token_ids.shape)}")

    token_ids = token_ids.to(lm.device)
    outputs = lm.model(input_ids=token_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    n_layers_total = len(hidden_states)
    if not -n_layers_total <= layer_idx < n_layers_total:
        raise IndexError(
            f"layer_idx={layer_idx} fuera de rango; hidden_states tiene "
            f"{n_layers_total} entradas (rango válido: "
            f"[{-n_layers_total}, {n_layers_total - 1}])"
        )
    hidden: torch.Tensor = hidden_states[layer_idx]
    return _drift_from_hidden(hidden)


@torch.no_grad()
def activation_drift_multilayer(
    lm: LoadedModel,
    token_ids: torch.Tensor,
    layer_indices: Sequence[int],
) -> dict[int, torch.Tensor]:
    """
    Variante eficiente: deriva por token en varias capas en **un único forward
    pass**. Necesario para exp_003 (barrido sobre {0, 4, 8, 12, 16, 20, 23} en
    Pythia 1.4B): si se llamara a `activation_drift_last_layer` por capa, se
    pagarían N forwards redundantes.

    Args:
        lm: modelo cargado; debe haberse cargado con output_hidden_states=True.
        token_ids: tensor (1, L).
        layer_indices: índices de capa a extraer; admite negativos.

    Returns:
        dict {layer_idx_original: tensor (L-1,) float32} con la deriva por
        capa solicitada, preservando el signo del índice tal cual lo pasó el
        caller (para que el llamador pueda re-leer su propia configuración).
    """
    if token_ids.dim() != 2 or token_ids.shape[0] != 1:
        raise ValueError(f"token_ids debe tener shape (1, L), recibido {tuple(token_ids.shape)}")
    if len(layer_indices) == 0:
        raise ValueError("layer_indices vacío")

    token_ids = token_ids.to(lm.device)
    outputs = lm.model(input_ids=token_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    n_layers_total = len(hidden_states)

    drifts: dict[int, torch.Tensor] = {}
    for li in layer_indices:
        if not -n_layers_total <= li < n_layers_total:
            raise IndexError(
                f"layer_idx={li} fuera de rango; hidden_states tiene "
                f"{n_layers_total} entradas"
            )
        drifts[li] = _drift_from_hidden(hidden_states[li])
    return drifts
