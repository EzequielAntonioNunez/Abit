from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import torch
from tqdm import tqdm

from iat.datasets import load_wikitext
from iat.io import make_run_dir, read_yaml, snapshot_run, write_json
from iat.metrics import compute_m1_cf, compute_m1_marginal
from iat.models import load_model
from iat.seeding import set_seed
from iat.stats import spearman_with_bootstrap_ci

EXPERIMENT_DIR = Path(__file__).parent


def _pg_unpack(row: pd.Series) -> dict[str, float | int]:
    ci = list(row["CI95"])
    return {
        "rho": float(row["r"]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p_value": float(row["p_val"]),
        "n": int(row["n"]),
    }


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
    k_block = int(config["m1_cf"]["k"])
    top_k = int(config["m1_marginal"]["top_k"])
    kl_dtype = getattr(torch, config["precision"]["kl_dtype"])

    records: list[dict[str, object]] = []

    for doc_idx, text in enumerate(tqdm(texts, desc="docs")):
        enc = lm.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_ctx)
        ids = enc.input_ids
        seq_len = ids.shape[1]
        if seq_len < min_ctx:
            continue
        max_target = seq_len - 1 - k_block
        if max_target < skip:
            continue
        positions = list(range(skip, max_target + 1))

        rows_cf = compute_m1_cf(lm, ids, positions=positions, k=k_block, dtype_kl=kl_dtype)
        rows_marg = compute_m1_marginal(
            lm, ids, positions=positions, top_k=top_k, dtype_kl=kl_dtype
        )

        by_pos_marg = {int(r["target_pos"]): r for r in rows_marg}
        for r_cf in rows_cf:
            pos = int(r_cf["target_pos"])
            r_m = by_pos_marg[pos]
            records.append(
                {
                    "doc_idx": doc_idx,
                    "position": pos,
                    "token_id": int(ids[0, pos].item()),
                    "surprisal_bits": float(r_cf["surprisal_bits"]),
                    "v_top": int(r_cf["v_top"]),
                    "m1_cf_kl_nats": float(r_cf["m1_cf_kl_nats"]),
                    "delta_cloze_cf": float(r_cf["delta_cloze_cf"]),
                    "m1_marginal_kl_nats": float(r_m["m1_marginal_kl_nats"]),
                    "t_i_in_topk": int(r_m["t_i_in_topk"]),
                    "top_k_mass_pre_renorm": float(r_m["top_k_mass_pre_renorm"]),
                }
            )

    df = pd.DataFrame.from_records(records)
    counts = df["token_id"].value_counts()
    df["log_freq"] = np.log(df["token_id"].map(counts).to_numpy())

    if config["output"]["write_records"]:
        df.to_parquet(run_dir / "records.parquet", index=False)

    n_boot = int(config["bootstrap"]["n_boot"])
    alpha = float(config["bootstrap"]["alpha"])
    seed = int(config["seed"])

    corr_p1 = spearman_with_bootstrap_ci(
        df["m1_marginal_kl_nats"].to_numpy(),
        df["surprisal_bits"].to_numpy(),
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )
    corr_p2 = spearman_with_bootstrap_ci(
        df["m1_cf_kl_nats"].to_numpy(),
        df["surprisal_bits"].to_numpy(),
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )
    corr_p3 = spearman_with_bootstrap_ci(
        df["m1_marginal_kl_nats"].to_numpy(),
        df["m1_cf_kl_nats"].to_numpy(),
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )

    partial_p4 = pg.partial_corr(
        data=df,
        x="m1_marginal_kl_nats",
        y="delta_cloze_cf",
        covar="surprisal_bits",
        method="spearman",
    ).iloc[0]
    partial_p5 = pg.partial_corr(
        data=df,
        x="m1_cf_kl_nats",
        y="delta_cloze_cf",
        covar="surprisal_bits",
        method="spearman",
    ).iloc[0]
    partial_marg_vs_logfreq = pg.partial_corr(
        data=df,
        x="m1_marginal_kl_nats",
        y="surprisal_bits",
        covar="log_freq",
        method="spearman",
    ).iloc[0]
    partial_cf_vs_logfreq = pg.partial_corr(
        data=df,
        x="m1_cf_kl_nats",
        y="surprisal_bits",
        covar="log_freq",
        method="spearman",
    ).iloc[0]

    summary = {
        "experiment": config["experiment"]["id"],
        "model": lm.name,
        "device": str(lm.device),
        "dtype": str(lm.dtype),
        "kl_dtype": str(kl_dtype),
        "k_block": k_block,
        "top_k": top_k,
        "n_docs_used": int(df["doc_idx"].nunique()),
        "n_tokens": len(df),
        "fraction_t_i_in_topk": float(df["t_i_in_topk"].mean()),
        "top_k_mass_pre_renorm_mean": float(df["top_k_mass_pre_renorm"].mean()),
        "descriptives": {
            "surprisal_bits": {
                "mean": float(df["surprisal_bits"].mean()),
                "median": float(df["surprisal_bits"].median()),
                "p95": float(np.percentile(df["surprisal_bits"], 95)),
            },
            "m1_marginal_kl_nats": {
                "mean": float(df["m1_marginal_kl_nats"].mean()),
                "median": float(df["m1_marginal_kl_nats"].median()),
                "p95": float(np.percentile(df["m1_marginal_kl_nats"], 95)),
            },
            "m1_cf_kl_nats": {
                "mean": float(df["m1_cf_kl_nats"].mean()),
                "median": float(df["m1_cf_kl_nats"].median()),
                "p95": float(np.percentile(df["m1_cf_kl_nats"], 95)),
            },
            "delta_cloze_cf": {
                "mean": float(df["delta_cloze_cf"].mean()),
                "median": float(df["delta_cloze_cf"].median()),
                "p95": float(np.percentile(df["delta_cloze_cf"], 95)),
            },
        },
        "P1_sanity_marginal_vs_shannon": {
            "rho": corr_p1.coefficient,
            "ci_low": corr_p1.ci_low,
            "ci_high": corr_p1.ci_high,
            "p_value": corr_p1.p_value,
        },
        "P2_sanity_cf_vs_shannon": {
            "rho": corr_p2.coefficient,
            "ci_low": corr_p2.ci_low,
            "ci_high": corr_p2.ci_high,
            "p_value": corr_p2.p_value,
        },
        "P3_marginal_vs_cf_coherence": {
            "rho": corr_p3.coefficient,
            "ci_low": corr_p3.ci_low,
            "ci_high": corr_p3.ci_high,
            "p_value": corr_p3.p_value,
        },
        "P4_partial_marginal_cloze_given_shannon": _pg_unpack(partial_p4),
        "P5_partial_cf_cloze_given_shannon": _pg_unpack(partial_p5),
        "control_partial_marginal_shannon_given_logfreq": _pg_unpack(partial_marg_vs_logfreq),
        "control_partial_cf_shannon_given_logfreq": _pg_unpack(partial_cf_vs_logfreq),
    }
    if config["output"]["write_summary"]:
        write_json(run_dir / "summary.json", summary)

    print("\n=== exp_004b summary ===")
    print(f"  n_tokens: {summary['n_tokens']}, docs: {summary['n_docs_used']}")
    print(f"  k_block={k_block}, top_k={top_k}, kl_dtype={kl_dtype}")
    print(f"  fraction t_i in top_k: {summary['fraction_t_i_in_topk']:.3f}")
    print(f"  top_k mass (pre-renorm) mean: {summary['top_k_mass_pre_renorm_mean']:.3f}")
    print()
    print("  PRIMARIAS (sanity):")
    print(
        f"    P1 rho(M1_marginal, Shannon)    = {corr_p1.coefficient:+.4f}  "
        f"[{corr_p1.ci_low:+.4f}, {corr_p1.ci_high:+.4f}]"
    )
    print(
        f"    P2 rho(M1_cf, Shannon)          = {corr_p2.coefficient:+.4f}  "
        f"[{corr_p2.ci_low:+.4f}, {corr_p2.ci_high:+.4f}]"
    )
    print("  SECUNDARIA (coherencia):")
    print(
        f"    P3 rho(M1_marginal, M1_cf)      = {corr_p3.coefficient:+.4f}  "
        f"[{corr_p3.ci_low:+.4f}, {corr_p3.ci_high:+.4f}]"
    )
    print("  PRINCIPALES DE TEORIA:")
    print(
        f"    P4 partial(M1_marginal, cloze | Shannon) = "
        f"{float(partial_p4['r']):+.4f}  IC{list(partial_p4['CI95'])}"
    )
    print(
        f"    P5 partial(M1_cf, cloze | Shannon)       = "
        f"{float(partial_p5['r']):+.4f}  IC{list(partial_p5['CI95'])}"
    )
    print(f"\nresultados en: {run_dir}")


if __name__ == "__main__":
    main()
