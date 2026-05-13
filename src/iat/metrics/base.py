from __future__ import annotations

from typing import Protocol

import torch

from iat.models import LoadedModel


class TokenMetric(Protocol):
    """
    Protocolo para métricas que asignan un valor escalar a cada token en una secuencia.
    """

    name: str

    def __call__(self, lm: LoadedModel, token_ids: torch.Tensor) -> torch.Tensor: ...
