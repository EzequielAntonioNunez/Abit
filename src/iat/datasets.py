from __future__ import annotations

from datasets import load_dataset


def load_wikitext(
    split: str = "validation",
    min_chars: int = 200,
    max_docs: int | None = None,
) -> list[str]:
    """
    Carga WikiText-103 y filtra documentos vacíos o demasiado cortos.

    Devuelve lista de textos crudos en orden original (determinista).
    """
    ds = load_dataset("wikitext", "wikitext-103-v1", split=split)
    texts = [t for t in ds["text"] if len(t) >= min_chars]
    if max_docs is not None:
        texts = texts[:max_docs]
    return texts
