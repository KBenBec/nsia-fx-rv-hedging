# -*- coding: utf-8 -*-
"""
Risk metrics: VaR / ES + drawdown + basic performance stats.
"""

from __future__ import annotations
import numpy as np


def var_es(pnl: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    """
    Historical VaR/ES on pnl distribution (loss = -pnl).
    Returns (VaR, ES) as positive numbers (losses).
    """
    pnl = np.asarray(pnl, dtype=float)
    loss = -pnl
    q = np.quantile(loss, alpha)
    tail = loss[loss >= q]
    es = float(np.mean(tail)) if len(tail) > 0 else float(q)
    return float(q), float(es)


def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=float)
    peak = -np.inf
    mdd = 0.0
    for x in equity:
        peak = max(peak, x)
        dd = peak - x
        mdd = max(mdd, dd)
    return float(mdd)


def sharpe(pnl: np.ndarray, eps: float = 1e-12) -> float:
    pnl = np.asarray(pnl, dtype=float)
    mu = float(np.mean(pnl))
    sd = float(np.std(pnl, ddof=1)) + eps
    return float(mu / sd)


def summary_stats(pnl: np.ndarray, equity: np.ndarray) -> dict:
    pnl = np.asarray(pnl, dtype=float)
    equity = np.asarray(equity, dtype=float)
    v, e = var_es(pnl, alpha=0.99)
    return {
        "mean_pnl": float(np.mean(pnl)),
        "std_pnl": float(np.std(pnl, ddof=1)),
        "sharpe_like": sharpe(pnl),
        "var_99": v,
        "es_99": e,
        "max_drawdown": max_drawdown(equity),
        "total_pnl": float(equity[-1]) if len(equity) else 0.0,
    }
