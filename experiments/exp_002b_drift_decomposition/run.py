from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.api as sm

from iat.io import make_run_dir, read_yaml, snapshot_run, write_json
from iat.seeding import set_seed

EXPERIMENT_DIR = Path(__file__).parent


def _find_latest_source_records(source_dir: Path, glob: str) -> Path:
    candidates = sorted(source_dir.glob(glob))
    if not candidates:
        raise FileNotFoundError(
            f"no se encontro ningun records.parquet en {source_dir / glob}; "
            "ejecuta primero el experimento fuente"
        )
    return candidates[-1]


def _bootstrap_mean_ci(
    x: np.ndarray, n_boot: int, alpha: float, rng: np.random.Generator
) -> tuple[float, float]:
    n = x.size
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boot[i] = x[rng.integers(0, n, size=n)].mean()
    low, high = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def main() -> None:
    config = read_yaml(EXPERIMENT_DIR / "config.yaml")
    set_seed(config["seed"])

    run_dir = make_run_dir(EXPERIMENT_DIR)
    snapshot_run(run_dir, config)
    figdir = run_dir / "figures"
    figdir.mkdir(exist_ok=True)

    source_dir = EXPERIMENT_DIR.parent / config["input"]["source_experiment_dir"]
    records_path = _find_latest_source_records(source_dir, config["input"]["records_glob"])

    df = pd.read_parquet(records_path)
    n_total = len(df)

    counts = df["token_id"].value_counts()
    df["token_count"] = df["token_id"].map(counts).astype(np.int64)
    df["log_freq"] = np.log(df["token_count"].to_numpy())

    n_deciles = int(config["analysis"]["n_deciles"])
    n_boot = int(config["analysis"]["bootstrap"]["n_boot"])
    alpha = float(config["analysis"]["bootstrap"]["alpha"])

    df["surprisal_decile"] = pd.qcut(
        df["surprisal_bits"], n_deciles, labels=False, duplicates="drop"
    ).astype("Int64")
    df["logfreq_decile"] = pd.qcut(
        df["log_freq"], n_deciles, labels=False, duplicates="drop"
    ).astype("Int64")

    n_surp_bins = int(df["surprisal_decile"].nunique())
    n_freq_bins = int(df["logfreq_decile"].nunique())

    partial = pg.partial_corr(
        data=df,
        x="surprisal_bits",
        y="activation_drift_last",
        covar="log_freq",
        method="spearman",
    ).iloc[0]
    rho_partial = float(partial["r"])
    partial_ci = list(partial["CI95"])
    ci_lo, ci_hi = float(partial_ci[0]), float(partial_ci[1])
    pval_partial = float(partial["p_val"])

    rho_marginal_pg = pg.corr(
        x=df["surprisal_bits"], y=df["activation_drift_last"], method="spearman"
    ).iloc[0]
    rho_marginal = float(rho_marginal_pg["r"])

    rng = np.random.default_rng(config["seed"])
    curve: list[dict[str, object]] = []
    for dec in sorted(df["surprisal_decile"].dropna().unique().tolist()):
        sub = df.loc[df["surprisal_decile"] == dec, "activation_drift_last"].to_numpy()
        lo, hi = _bootstrap_mean_ci(sub, n_boot=n_boot, alpha=alpha, rng=rng)
        curve.append(
            {
                "decile": int(dec),
                "mean": float(sub.mean()),
                "median": float(np.median(sub)),
                "ci_low": lo,
                "ci_high": hi,
                "n": int(sub.size),
            }
        )

    stratified: list[dict[str, object]] = []
    for fdec in sorted(df["logfreq_decile"].dropna().unique().tolist()):
        for sdec in sorted(df["surprisal_decile"].dropna().unique().tolist()):
            mask = (df["logfreq_decile"] == fdec) & (df["surprisal_decile"] == sdec)
            sub = df.loc[mask, "activation_drift_last"].to_numpy()
            if sub.size == 0:
                continue
            stratified.append(
                {
                    "logfreq_decile": int(fdec),
                    "surprisal_decile": int(sdec),
                    "mean": float(sub.mean()),
                    "n": int(sub.size),
                }
            )

    design = pd.DataFrame(
        {
            "const": 1.0,
            "surprisal": df["surprisal_bits"].to_numpy(),
            "log_freq": df["log_freq"].to_numpy(),
            "surp_x_logfreq": (df["surprisal_bits"] * df["log_freq"]).to_numpy(),
        }
    )
    y = df["activation_drift_last"].to_numpy()
    ols = sm.OLS(y, design).fit()
    ols_params = {k: float(v) for k, v in ols.params.items()}
    ols_conf = {k: [float(lo), float(hi)] for k, (lo, hi) in ols.conf_int().iterrows()}
    ols_p = {k: float(v) for k, v in ols.pvalues.items()}

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    hb = ax.hexbin(
        df["surprisal_bits"], df["activation_drift_last"], gridsize=40, bins="log", cmap="viridis"
    )
    fig.colorbar(hb, ax=ax, label="log10(N)")
    ax.set_xlabel("surprisal Shannon (bits)")
    ax.set_ylabel("activation drift last (||Δh||/√d)")
    ax.set_title(f"Distribucion bivariante (N={n_total})")
    fig.tight_layout()
    fig.savefig(figdir / "bivariate_hexbin.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    means = np.array([c["mean"] for c in curve])
    lows = np.array([c["ci_low"] for c in curve])
    highs = np.array([c["ci_high"] for c in curve])
    deciles = np.array([c["decile"] for c in curve])
    ci_label = f"IC {int((1 - alpha) * 100)}% bootstrap"
    ax.fill_between(deciles, lows, highs, alpha=0.25, label=ci_label)
    ax.plot(deciles, means, marker="o", label="drift media")
    ax.set_xlabel("decil de surprisal (0 = bajo, 9 = alto)")
    ax.set_ylabel("activation drift last (media)")
    ax.set_title("Curva drift por decil de surprisal (marginal)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "drift_by_surprisal_decile.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    strat_df = pd.DataFrame(stratified)
    cmap = plt.colormaps["viridis"]
    fdecs = sorted(strat_df["logfreq_decile"].unique().tolist())
    for i, fdec in enumerate(fdecs):
        sub = strat_df[strat_df["logfreq_decile"] == fdec].sort_values("surprisal_decile")
        color = cmap(i / max(len(fdecs) - 1, 1))
        ax.plot(
            sub["surprisal_decile"],
            sub["mean"],
            marker=".",
            color=color,
            alpha=0.85,
            label=f"log_freq dec {fdec}",
        )
    ax.set_xlabel("decil de surprisal")
    ax.set_ylabel("drift media")
    ax.set_title("Curva drift por decil de surprisal, estratificada por decil log_freq")
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figdir / "drift_by_surprisal_stratified.png", dpi=150)
    plt.close(fig)

    residuals = ols.resid
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.scatter(ols.fittedvalues, residuals, s=4, alpha=0.3)
    ax.axhline(0.0, color="k", lw=0.7)
    ax.set_xlabel("drift predicho por OLS")
    ax.set_ylabel("residual")
    ax.set_title("Residuales OLS (drift ~ surprisal + log_freq + interaccion)")
    fig.tight_layout()
    fig.savefig(figdir / "ols_residuals.png", dpi=150)
    plt.close(fig)

    n_unique_tokens = int(df["token_id"].nunique())
    singletons = int((df["token_count"] == 1).sum())

    summary: dict[str, object] = {
        "experiment": config["experiment"]["id"],
        "source_records": str(records_path),
        "n_tokens": n_total,
        "n_unique_token_ids": n_unique_tokens,
        "n_singletons_in_sample": singletons,
        "n_surprisal_bins": n_surp_bins,
        "n_logfreq_bins": n_freq_bins,
        "spearman_marginal_surprisal_drift": rho_marginal,
        "spearman_partial_surprisal_drift_given_logfreq": {
            "rho": rho_partial,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "p_value": pval_partial,
            "n": int(partial["n"]),
        },
        "drift_by_surprisal_decile_marginal": curve,
        "drift_by_surprisal_decile_stratified_by_logfreq": stratified,
        "ols_drift_on_surprisal_logfreq_interaction": {
            "params": ols_params,
            "ci95": ols_conf,
            "p_values": ols_p,
            "r_squared": float(ols.rsquared),
            "n": int(ols.nobs),
        },
    }
    if config["output"]["write_summary"]:
        write_json(run_dir / "summary.json", summary)

    print("\n=== exp_002b summary ===")
    print(f"  source: {records_path}")
    print(f"  n_tokens: {n_total}, n_unique: {n_unique_tokens}, singletons: {singletons}")
    print(f"  spearman marginal: {rho_marginal:.4f}")
    print(
        f"  spearman parcial | log_freq: {rho_partial:.4f} "
        f"(IC95 [{ci_lo:.4f}, {ci_hi:.4f}], p={pval_partial:.3g})"
    )
    print(f"  OLS R^2: {ols.rsquared:.4f}")
    print("  OLS coef (CI95, p):")
    for k in ols_params:
        lo, hi = ols_conf[k]
        print(f"    {k}: {ols_params[k]:+.5f}  [{lo:+.5f}, {hi:+.5f}]  p={ols_p[k]:.3g}")
    print(f"\nresultados en: {run_dir}")


if __name__ == "__main__":
    main()
