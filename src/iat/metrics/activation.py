from __future__ import annotations

import math

import torch

from iat.models import LoadedModel


@torch.no_grad()
def activation_drift_last_layer(lm: LoadedModel, token_ids: torch.Tensor) -> torch.Tensor:
    """
    M2_last: || h_L(t_1..t_i) - h_L(t_1..t_{i-1}) ||_2 / sqrt(d_L) por token.

    Por causalidad del masking en LMs autoregresivos, ambos estados se obtienen
    en un solo forward pass: hidden_states[-1] tiene shape (1, L, d_L) y la fila
    i corresponde al estado tras consumir tokens t_1..t_i.

    Args:
        lm: modelo cargado; debe haberse cargado con output_hidden_states=True.
        token_ids: tensor (1, L) con ids de tokens.

    Returns:
        tensor (L-1,) en float32 con la deriva por token. La posición i del
        output corresponde al token original i+1 (alineado con
        shannon_surprisal para permitir comparación token-a-token).
    """
    if token_ids.dim() != 2 or token_ids.shape[0] != 1:
        raise ValueError(f"token_ids debe tener shape (1, L), recibido {tuple(token_ids.shape)}")

    token_ids = token_ids.to(lm.device)
    outputs = lm.model(input_ids=token_ids, output_hidden_states=True)
    last_hidden: torch.Tensor = outputs.hidden_states[-1]
    h_last = last_hidden[0].float()  # (L, d)
    diff = h_last[1:] - h_last[:-1]  # (L-1, d)
    d = h_last.shape[-1]
    drift: torch.Tensor = diff.norm(dim=-1) / math.sqrt(d)
    return drift
