# -*- coding: utf-8 -*-
"""
Calibration (multi-start style) for simplified Regime-SVJ model.
This is NOT a production-grade HMM calibration; it's a robust public version:
- regime proxy via EWM-vol split
- jump detection via robust threshold
- estimate p11/p22 via empirical transitions
- estimate sigma1/sigma2 from regime subsets
- estimate jump intensity & jump distribution from flagged points
Then do a small multi-start refinement on (lam, mu_j, sigma_j) by grid search.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .fx_regime_svj import (
    SVJParams,
    regime_proxy_from_ewm_vol,
    detect_jumps_threshold,
    loglik_svj_gaussian_mixture,
    simulate_regimes,
)

@dataclass
class CalibResult:
    params: SVJParams
    reg: np.ndarray
    ll: float


def estimate_transition_probs(reg: np.ndarray) -> tuple[float, float]:
    reg = np.asarray(reg, dtype=int)
    # transitions counts
    c11 = c12 = c21 = c22 = 0
    for t in range(1, len(reg)):
        a, b = reg[t-1], reg[t]
        if a == 1 and b == 1: c11 += 1
        if a == 1 and b == 2: c12 += 1
        if a == 2 and b == 1: c21 += 1
        if a == 2 and b == 2: c22 += 1
    p11 = (c11 + 1) / (c11 + c12 + 2)  # Laplace smoothing
    p22 = (c22 + 1) / (c21 + c22 + 2)
    return float(p11), float(p22)


def _safe_std(x: np.ndarray) -> float:
    s = float(np.std(x, ddof=1)) if len(x) > 1 else float(np.std(x))
    return max(s, 1e-6)


def calibrate_from_returns(
    r: np.ndarray,
    span_vol: int = 20,
    jump_k: float = 4.0,
    multistart_grid: dict | None = None,
    seed: int = 0,
) -> CalibResult:
    """
    Main calibration entrypoint.
    """
    r = np.asarray(r, dtype=float)
    reg = regime_proxy_from_ewm_vol(r, span=span_vol)

    # jump flags (robust)
    jflag = detect_jumps_threshold(r, k=jump_k)

    # estimate transition
    p11, p22 = estimate_transition_probs(reg)

    # estimate sigmas excluding jump points (more stable)
    r_clean = r[~jflag]
    reg_clean = reg[~jflag]
    sigma1 = _safe_std(r_clean[reg_clean == 1])
    sigma2 = _safe_std(r_clean[reg_clean == 2])

    # estimate jump distribution from flagged points after removing base noise
    r_jump = r[jflag]
    if len(r_jump) >= 5:
        mu_j0 = float(np.mean(r_jump))
        sigma_j0 = _safe_std(r_jump - mu_j0)
        lam0 = float(len(r_jump) / len(r))
    else:
        mu_j0, sigma_j0, lam0 = 0.0, 0.01, 0.01

    # multi-start refinement (tiny grid) for (lam, mu_j, sigma_j)
    if multistart_grid is None:
        multistart_grid = {
            "lam": [max(0.001, 0.5 * lam0), lam0, min(0.20, 2.0 * lam0)],
            "mu_j": [0.5 * mu_j0, mu_j0, 1.5 * mu_j0],
            "sigma_j": [max(0.002, 0.5 * sigma_j0), sigma_j0, 2.0 * sigma_j0],
        }

    best = None
    best_ll = -np.inf

    for lam in multistart_grid["lam"]:
        for mu_j in multistart_grid["mu_j"]:
            for sigma_j in multistart_grid["sigma_j"]:
                params = SVJParams(p11=p11, p22=p22, sigma1=sigma1, sigma2=sigma2,
                                   lam=float(max(lam, 1e-6)), mu_j=float(mu_j),
                                   sigma_j=float(max(sigma_j, 1e-6)))
                ll = loglik_svj_gaussian_mixture(r, reg, params)
                if ll > best_ll:
                    best_ll = ll
                    best = params

    assert best is not None
    return CalibResult(params=best, reg=reg, ll=float(best_ll))


def load_fx_csv(path: str, price_col: str = "rate") -> pd.Series:
    """
    Expected CSV format (synthetic/public):
    columns: date, pair, rate
    """
    df = pd.read_csv(path)
    if price_col not in df.columns:
        raise ValueError(f"Missing '{price_col}' column in {path}. Found: {df.columns.tolist()}")
    s = pd.Series(df[price_col].values, index=pd.to_datetime(df["date"]))
    s.name = price_col
    return s


def log_returns_from_prices(px: np.ndarray) -> np.ndarray:
    px = np.asarray(px, dtype=float)
    px = np.maximum(px, 1e-12)
    r = np.diff(np.log(px))
    return r
