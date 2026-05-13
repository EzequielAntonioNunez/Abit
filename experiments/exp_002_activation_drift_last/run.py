from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from iat.datasets import load_wikitext
from iat.io import make_run_dir, read_yaml, snapshot_run, write_json
from iat.metrics import activation_drift_last_layer, shannon_surprisal
from iat.models import load_model
from iat.seeding import set_seed
from iat.stats import spearman_with_bootstrap_ci

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

    for doc_idx, text in enumerate(tqdm(texts, desc="docs")):
        enc = lm.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_ctx)
        ids = enc.input_ids
        if ids.shape[1] < min_ctx:
            continue

        surprisal = shannon_surprisal(lm, ids).cpu().numpy()
        drift = activation_drift_last_layer(lm, ids).cpu().numpy()

        usable_s = surprisal[skip:]
        usable_d = drift[skip:]
        assert usable_s.shape == usable_d.shape

        for offset, (bits, dval) in enumerate(zip(usable_s, usable_d, strict=True)):
            position = skip + offset + 1
            token_id = int(ids[0, position].item())
            records.append(
                {
                    "doc_idx": doc_idx,
                    "position": position,
                    "token_id": token_id,
                    "surprisal_bits": float(bits),
                    "activation_drift_last": float(dval),
                }
            )

    df = pd.DataFrame.from_records(records)
    if config["output"]["write_records"]:
        df.to_parquet(run_dir / "records.parquet", index=False)

    corr = spearman_with_bootstrap_ci(
        df["surprisal_bits"].to_numpy(),
        df["activation_drift_last"].to_numpy(),
        n_boot=config["bootstrap"]["n_boot"],
        alpha=config["bootstrap"]["alpha"],
        seed=config["seed"],
    )

    summary = {
        "experiment": config["experiment"]["id"],
        "model": lm.name,
        "device": str(lm.device),
        "dtype": str(lm.dtype),
        "n_docs_used": int(df["doc_idx"].nunique()),
        "n_tokens": len(df),
        "surprisal_bits": {
            "mean": float(df["surprisal_bits"].mean()),
            "median": float(df["surprisal_bits"].median()),
            "p95": float(np.percentile(df["surprisal_bits"], 95)),
        },
        "activation_drift_last": {
            "mean": float(df["activation_drift_last"].mean()),
            "median": float(df["activation_drift_last"].median()),
            "p95": float(np.percentile(df["activation_drift_last"], 95)),
        },
        "spearman_surprisal_vs_drift": {
            "rho": corr.coefficient,
            "p_value": corr.p_value,
            "n": corr.n,
            "ci_low": corr.ci_low,
            "ci_high": corr.ci_high,
            "alpha": config["bootstrap"]["alpha"],
            "n_boot": config["bootstrap"]["n_boot"],
        },
    }
    if config["output"]["write_summary"]:
        write_json(run_dir / "summary.json", summary)

    print("\n=== exp_002 summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nresultados en: {run_dir}")


if __name__ == "__main__":
    main()
