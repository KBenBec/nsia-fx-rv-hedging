# -*- coding: utf-8 -*-
"""
Regime-switching SVJ (simplified) for FX log-returns.
- 2 regimes for volatility with Markov switching
- Gaussian jumps (Poisson intensity) added to returns
This is a PUBLIC/EDUCATIONAL version with synthetic-data friendly design.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SVJParams:
    # Regime transition matrix
    p11: float  # P(reg=1 -> 1)
    p22: float  # P(reg=2 -> 2)

    # Regime vols (annualized-like scale but here per-step)
    sigma1: float
    sigma2: float

    # Jump component (per-step)
    lam: float      # Poisson intensity per step
    mu_j: float     # mean jump size (in log-return)
    sigma_j: float  # std jump size


def _clip_prob(x: float) -> float:
    return float(np.clip(x, 1e-6, 1 - 1e-6))


def simulate_regimes(n: int, p11: float, p22: float, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate 2-state Markov chain regimes in {1,2}.
    """
    p11 = _clip_prob(p11)
    p22 = _clip_prob(p22)
    P = np.array([[p11, 1 - p11],
                  [1 - p22, p22]], dtype=float)

    reg = np.empty(n, dtype=int)
    reg[0] = 1
    for t in range(1, n):
        prev = reg[t - 1] - 1
        reg[t] = 1 + rng.choice([0, 1], p=P[prev])
    return reg


def simulate_svj_returns(n: int, params: SVJParams, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate log-returns r_t = sigma_reg * eps + jump_t, eps~N(0,1), jump_t = sum_{k=1..N_t} J_k.
    """
    reg = simulate_regimes(n, params.p11, params.p22, rng)
    sig = np.where(reg == 1, params.sigma1, params.sigma2)

    eps = rng.standard_normal(n)
    base = sig * eps

    # jumps
    Nj = rng.poisson(params.lam, size=n)
    jump = np.zeros(n, dtype=float)
    for t in range(n):
        if Nj[t] > 0:
            jump[t] = rng.normal(params.mu_j, params.sigma_j, size=Nj[t]).sum()

    return base + jump


def loglik_svj_gaussian_mixture(r: np.ndarray, reg: np.ndarray, params: SVJParams) -> float:
    """
    Approximate log-likelihood given regimes.
    For each step: r = Normal(0, sigma_reg^2) convolved with compound Poisson Gaussian.
    Approximation: mixture over N_j in {0,1,2} (truncated).
    """
    r = np.asarray(r, dtype=float)
    reg = np.asarray(reg, dtype=int)
    sig = np.where(reg == 1, params.sigma1, params.sigma2)

    # Truncated mixture for jumps: N=0,1,2
    lam = max(params.lam, 1e-10)
    w0 = np.exp(-lam)
    w1 = w0 * lam
    w2 = w0 * (lam ** 2) / 2.0
    weights = np.array([w0, w1, w2], dtype=float)
    weights = weights / weights.sum()

    # For N jumps, jump distribution ~ Normal(N*mu_j, N*sigma_j^2)
    # Convolution with base Normal gives Normal(mean, var).
    ll = 0.0
    for t in range(len(r)):
        dens = 0.0
        for N, w in enumerate(weights):
            mean = N * params.mu_j
            var = sig[t] ** 2 + N * (params.sigma_j ** 2)
            var = max(var, 1e-12)
            dens += w * (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * ((r[t] - mean) ** 2) / var)
        ll += np.log(max(dens, 1e-300))
    return float(ll)


def detect_jumps_threshold(r: np.ndarray, k: float = 4.0) -> np.ndarray:
    """
    Simple jump proxy: |r_t| > k * MAD(r) (robust).
    Returns boolean array.
    """
    r = np.asarray(r, dtype=float)
    med = np.median(r)
    mad = np.median(np.abs(r - med)) + 1e-12
    z = np.abs(r - med) / (1.4826 * mad)
    return z > k


def regime_proxy_from_ewm_vol(r: np.ndarray, span: int = 20) -> np.ndarray:
    """
    Regime proxy using EWM volatility:
    - compute ewm std
    - regime 2 if vol above median, else regime 1
    """
    r = np.asarray(r, dtype=float)
    alpha = 2.0 / (span + 1.0)
    v = np.zeros_like(r)
    m2 = 0.0
    for t in range(len(r)):
        m2 = (1 - alpha) * m2 + alpha * (r[t] ** 2)
        v[t] = np.sqrt(max(m2, 1e-12))
    thr = np.median(v)
    reg = np.where(v > thr, 2, 1)
    return reg.astype(int)
