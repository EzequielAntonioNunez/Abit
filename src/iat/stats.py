from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class CorrelationResult:
    method: str
    coefficient: float
    p_value: float
    n: int
    ci_low: float
    ci_high: float


def spearman_with_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> CorrelationResult:
    """
    Spearman con intervalo de confianza bootstrap.

    Devuelve coeficiente puntual, p-valor de la implementación de scipy
    e intervalo de confianza percentil bootstrap al (1-alpha).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError(f"x y y deben tener mismo shape; {x.shape} vs {y.shape}")
    if x.ndim != 1:
        raise ValueError(f"se esperaban arrays 1D; recibido {x.ndim}D")

    n = x.shape[0]
    rho, p = stats.spearmanr(x, y)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = stats.spearmanr(x[idx], y[idx]).statistic

    low, high = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CorrelationResult(
        method="spearman",
        coefficient=float(rho),
        p_value=float(p),
        n=n,
        ci_low=float(low),
        ci_high=float(high),
    )
