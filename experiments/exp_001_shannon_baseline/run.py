from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from iat.datasets import load_wikitext
from iat.io import make_run_dir, read_yaml, snapshot_run, write_json
from iat.metrics import shannon_surprisal
from iat.models import load_model
from iat.seeding import set_seed


EXPERIMENT_DIR = Path(__file__).parent


def main() -> None:
    config = read_yaml(EXPERIMENT_DIR / "config.yaml")
    set_seed(config["seed"])

    run_dir = make_run_dir(EXPERIMENT_DIR)
    snapshot_run(run_dir, config)

    lm = load_model(
        name=config["model"]["name"],
        revision=config["model"]["revision"],
        dtype=getattr(torch, config["model"]["dtype"]),
    )

    texts = load_wikitext(
        split=config["dataset"]["split"],
        min_chars=config["dataset"]["min_chars"],
        max_docs=config["dataset"]["max_docs"],
    )

    max_ctx = config["sampling"]["max_context_tokens"]
    min_ctx = config["sampling"]["min_context_tokens"]
    skip = config["sampling"]["warmup_skip_tokens"]

    records: list[dict[str, object]] = []
    nll_total_bits = 0.0
    tokens_total = 0

    for doc_idx, text in enumerate(tqdm(texts, desc="docs")):
        enc = lm.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_ctx)
        ids = enc.input_ids
        if ids.shape[1] < min_ctx:
            continue

        surprisal = shannon_surprisal(lm, ids).cpu().numpy()

        usable = surprisal[skip:]
        nll_total_bits += float(usable.sum())
        tokens_total += int(usable.size)

        for offset, bits in enumerate(usable):
            position = skip + offset + 1
            token_id = int(ids[0, position].item())
            records.append(
                {
                    "doc_idx": doc_idx,
                    "position": position,
                    "token_id": token_id,
                    "surprisal_bits": float(bits),
                }
            )

    avg_bits_per_token = nll_total_bits / max(tokens_total, 1)
    perplexity = float(2.0 ** avg_bits_per_token)

    if config["output"]["write_records"]:
        df = pd.DataFrame.from_records(records)
        df.to_parquet(run_dir / "records.parquet", index=False)

    summary = {
        "experiment": config["experiment"]["id"],
        "model": lm.name,
        "device": str(lm.device),
        "dtype": str(lm.dtype),
        "n_docs_used": int(np.unique([r["doc_idx"] for r in records]).size),
        "n_tokens": tokens_total,
        "avg_surprisal_bits": avg_bits_per_token,
        "perplexity": perplexity,
        "median_surprisal_bits": float(np.median([r["surprisal_bits"] for r in records])),
        "p95_surprisal_bits": float(np.percentile([r["surprisal_bits"] for r in records], 95)),
    }
    if config["output"]["write_summary"]:
        write_json(run_dir / "summary.json", summary)

    print("\n=== exp_001 summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nresultados en: {run_dir}")


if __name__ == "__main__":
    main()
