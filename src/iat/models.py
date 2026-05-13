from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass
class LoadedModel:
    name: str
    revision: str | None
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    dtype: torch.dtype

    @property
    def n_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    name: str,
    revision: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: Path | None = None,
    output_hidden_states: bool = True,
    output_attentions: bool = False,
) -> LoadedModel:
    """
    Carga un modelo causal y su tokenizer desde HuggingFace.

    revision: pin opcional al SHA o tag del modelo en HF para reproducibilidad.
    """
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        revision=revision,
        cache_dir=cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        name,
        revision=revision,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
    )
    model.to(device)  # type: ignore[arg-type]
    model.eval()  # type: ignore[no-untyped-call]
    return LoadedModel(
        name=name,
        revision=revision,
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
    )
