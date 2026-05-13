from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import torch
from tqdm import tqdm

from iat.datasets import load_wikitext
from iat.io import make_run_dir, read_yaml, snapshot_run, write_json
from iat.metrics import activation_drift_multilayer, shannon_surprisal
from iat.models import load_model
from iat.seeding import set_seed
from iat.stats import spearman_with_bootstrap_ci

EXPERIMENT_DIR = Path(__file__).parent


def _layer_col(idx: int) -> str:
    return f"drift_l{idx}"


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
    layer_indices: list[int] = list(config["layers"])

    records: list[dict[str, object]] = []

    for doc_idx, text in enumerate(tqdm(texts, desc="docs")):
        enc = lm.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_ctx)
        ids = enc.input_ids
        if ids.shape[1] < min_ctx:
            continue

        surprisal = shannon_surprisal(lm, ids).cpu().numpy()
        drifts = activation_drift_multilayer(lm, ids, layer_indices)
        drifts_np = {li: t.cpu().numpy() for li, t in drifts.items()}

        usable_s = surprisal[skip:]
        usable_d = {li: drifts_np[li][skip:] for li in layer_indices}
        for li in layer_indices:
            assert usable_d[li].shape == usable_s.shape

        for offset in range(usable_s.size):
            position = skip + offset + 1
            token_id = int(ids[0, position].item())
            row: dict[str, object] = {
                "doc_idx": doc_idx,
                "position": position,
                "token_id": token_id,
                "surprisal_bits": float(usable_s[offset]),
            }
            for li in layer_indices:
                row[_layer_col(li)] = float(usable_d[li][offset])
            records.append(row)

    df = pd.DataFrame.from_records(records)

    counts = df["token_id"].value_counts()
    df["log_freq"] = np.log(df["token_id"].map(counts).to_numpy())

    if config["output"]["write_records"]:
        df.to_parquet(run_dir / "records.parquet", index=False)

    per_layer: list[dict[str, object]] = []
    for li in layer_indices:
        col = _layer_col(li)
        corr = spearman_with_bootstrap_ci(
            df["surprisal_bits"].to_numpy(),
            df[col].to_numpy(),
            n_boot=int(config["bootstrap"]["n_boot"]),
            alpha=float(config["bootstrap"]["alpha"]),
            seed=int(config["seed"]),
        )
        partial = pg.partial_corr(
            data=df,
            x="surprisal_bits",
            y=col,
            covar="log_freq",
            method="spearman",
        ).iloc[0]
        partial_ci = list(partial["CI95"])
        per_layer.append(
            {
                "layer_idx": li,
                "drift_mean": float(df[col].mean()),
                "drift_median": float(df[col].median()),
                "drift_p95": float(np.percentile(df[col], 95)),
                "spearman_marginal_rho": corr.coefficient,
                "spearman_marginal_ci_low": corr.ci_low,
                "spearman_marginal_ci_high": corr.ci_high,
                "spearman_marginal_p_value": corr.p_value,
                "spearman_partial_rho": float(partial["r"]),
                "spearman_partial_ci_low": float(partial_ci[0]),
                "spearman_partial_ci_high": float(partial_ci[1]),
                "spearman_partial_p_value": float(partial["p_val"]),
                "n": int(corr.n),
            }
        )

    max_rho_marginal = max((p["spearman_marginal_rho"] for p in per_layer), default=float("nan"))
    max_rho_partial = max((p["spearman_partial_rho"] for p in per_layer), default=float("nan"))
    argmax_marginal = max(per_layer, key=lambda p: p["spearman_marginal_rho"])["layer_idx"]
    argmax_partial = max(per_layer, key=lambda p: p["spearman_partial_rho"])["layer_idx"]

    summary = {
        "experiment": config["experiment"]["id"],
        "model": lm.name,
        "device": str(lm.device),
        "dtype": str(lm.dtype),
        "n_docs_used": int(df["doc_idx"].nunique()),
        "n_tokens": len(df),
        "layers_swept": layer_indices,
        "per_layer": per_layer,
        "max_spearman_marginal": {"rho": max_rho_marginal, "layer_idx": argmax_marginal},
        "max_spearman_partial": {"rho": max_rho_partial, "layer_idx": argmax_partial},
    }
    if config["output"]["write_summary"]:
        write_json(run_dir / "summary.json", summary)

    print("\n=== exp_003 summary ===")
    for k, v in summary.items():
        if k == "per_layer":
            print("  per_layer:")
            for p in v:  # type: ignore[union-attr]
                print(
                    f"    l={p['layer_idx']:>3}  rho_marg={p['spearman_marginal_rho']:+.4f}  "
                    f"rho_part={p['spearman_partial_rho']:+.4f}  "
                    f"drift_mean={p['drift_mean']:.3f}"
                )
        else:
            print(f"  {k}: {v}")
    print(f"\nresultados en: {run_dir}")


if __name__ == "__main__":
    main()
