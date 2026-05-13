from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from iat.models import LoadedModel


@torch.no_grad()
def shannon_surprisal(lm: LoadedModel, token_ids: torch.Tensor) -> torch.Tensor:
    """
    Calcula -log_2 p(t_i | t_{<i}) para cada token a partir del segundo.

    Args:
        lm: modelo cargado
        token_ids: tensor (1, L) con ids de tokens

    Returns:
        tensor (L-1,) en bits. La posición i corresponde al token original i+1.
    """
    if token_ids.dim() != 2 or token_ids.shape[0] != 1:
        raise ValueError(f"token_ids debe tener shape (1, L), recibido {tuple(token_ids.shape)}")

    token_ids = token_ids.to(lm.device)
    outputs = lm.model(input_ids=token_ids)
    logits = outputs.logits[0, :-1, :].float()
    targets = token_ids[0, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    nll_nats = -log_probs[torch.arange(targets.size(0), device=lm.device), targets]
    return nll_nats / math.log(2.0)
