from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import torch
from tqdm import tqdm

from iat.datasets import load_wikitext
from iat.io import make_run_dir, read_yaml, snapshot_run, write_json
from iat.metrics import m1_and_cloze_block
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
    k = int(config["m1"]["k"])

    records: list[dict[str, object]] = []

    for doc_idx, text in enumerate(tqdm(texts, desc="docs")):
        enc = lm.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_ctx)
        ids = enc.input_ids
        seq_len = ids.shape[1]
        if seq_len < min_ctx:
            continue
        max_target = seq_len - 1 - k
        if max_target < skip:
            continue
        targets = list(range(skip, max_target + 1))

        rows = m1_and_cloze_block(lm, ids, targets=targets, k=k)
        for r in rows:
            i = int(r["target_pos"])
            records.append(
                {
                    "doc_idx": doc_idx,
                    "position": i,
                    "token_id": int(ids[0, i].item()),
                    "surprisal_bits": float(r["surprisal_bits"]),
                    "m1_kl_block_nats": float(r["m1_kl_block_nats"]),
                    "delta_cloze": float(r["delta_cloze"]),
                }
            )

    df = pd.DataFrame.from_records(records)
    counts = df["token_id"].value_counts()
    df["log_freq"] = np.log(df["token_id"].map(counts).to_numpy())

    if config["output"]["write_records"]:
        df.to_parquet(run_dir / "records.parquet", index=False)

    n_boot = int(config["bootstrap"]["n_boot"])
    alpha = float(config["bootstrap"]["alpha"])

    corr_m1_shannon = spearman_with_bootstrap_ci(
        df["m1_kl_block_nats"].to_numpy(),
        df["surprisal_bits"].to_numpy(),
        n_boot=n_boot,
        alpha=alpha,
        seed=int(config["seed"]),
    )
    corr_m1_cloze = spearman_with_bootstrap_ci(
        df["m1_kl_block_nats"].to_numpy(),
        df["delta_cloze"].to_numpy(),
        n_boot=n_boot,
        alpha=alpha,
        seed=int(config["seed"]),
    )
    corr_shannon_cloze = spearman_with_bootstrap_ci(
        df["surprisal_bits"].to_numpy(),
        df["delta_cloze"].to_numpy(),
        n_boot=n_boot,
        alpha=alpha,
        seed=int(config["seed"]),
    )

    partial_m1_cloze_given_shannon = pg.partial_corr(
        data=df,
        x="m1_kl_block_nats",
        y="delta_cloze",
        covar="surprisal_bits",
        method="spearman",
    ).iloc[0]
    partial_m1_shannon_given_logfreq = pg.partial_corr(
        data=df,
        x="m1_kl_block_nats",
        y="surprisal_bits",
        covar="log_freq",
        method="spearman",
    ).iloc[0]

    def _pg_unpack(row: pd.Series) -> dict[str, float | int]:
        ci = list(row["CI95"])
        return {
            "rho": float(row["r"]),
            "ci_low": float(ci[0]),
            "ci_high": float(ci[1]),
            "p_value": float(row["p_val"]),
            "n": int(row["n"]),
        }

    summary = {
        "experiment": config["experiment"]["id"],
        "model": lm.name,
        "device": str(lm.device),
        "dtype": str(lm.dtype),
        "k": k,
        "n_docs_used": int(df["doc_idx"].nunique()),
        "n_tokens": len(df),
        "descriptives": {
            "surprisal_bits": {
                "mean": float(df["surprisal_bits"].mean()),
                "median": float(df["surprisal_bits"].median()),
                "p95": float(np.percentile(df["surprisal_bits"], 95)),
            },
            "m1_kl_block_nats": {
                "mean": float(df["m1_kl_block_nats"].mean()),
                "median": float(df["m1_kl_block_nats"].median()),
                "p95": float(np.percentile(df["m1_kl_block_nats"], 95)),
            },
            "delta_cloze": {
                "mean": float(df["delta_cloze"].mean()),
                "median": float(df["delta_cloze"].median()),
                "p95": float(np.percentile(df["delta_cloze"], 95)),
            },
        },
        "spearman_m1_vs_shannon": {
            "rho": corr_m1_shannon.coefficient,
            "ci_low": corr_m1_shannon.ci_low,
            "ci_high": corr_m1_shannon.ci_high,
            "p_value": corr_m1_shannon.p_value,
            "n": int(corr_m1_shannon.n),
        },
        "spearman_m1_vs_cloze": {
            "rho": corr_m1_cloze.coefficient,
            "ci_low": corr_m1_cloze.ci_low,
            "ci_high": corr_m1_cloze.ci_high,
            "p_value": corr_m1_cloze.p_value,
        },
        "spearman_shannon_vs_cloze": {
            "rho": corr_shannon_cloze.coefficient,
            "ci_low": corr_shannon_cloze.ci_low,
            "ci_high": corr_shannon_cloze.ci_high,
            "p_value": corr_shannon_cloze.p_value,
        },
        "spearman_partial_m1_vs_cloze_given_shannon": _pg_unpack(
            partial_m1_cloze_given_shannon
        ),
        "spearman_partial_m1_vs_shannon_given_logfreq": _pg_unpack(
            partial_m1_shannon_given_logfreq
        ),
    }
    if config["output"]["write_summary"]:
        write_json(run_dir / "summary.json", summary)

    print("\n=== exp_004 summary ===")
    print(f"  n_tokens: {summary['n_tokens']}, docs: {summary['n_docs_used']}, k={k}")
    print(f"  rho(M1, Shannon) marginal = {corr_m1_shannon.coefficient:+.4f}")
    print(f"  rho(M1, cloze) marginal   = {corr_m1_cloze.coefficient:+.4f}")
    print(
        f"  rho(M1, cloze | Shannon)  = "
        f"{summary['spearman_partial_m1_vs_cloze_given_shannon']['rho']:+.4f}"
    )
    print(f"\nresultados en: {run_dir}")


if __name__ == "__main__":
    main()
