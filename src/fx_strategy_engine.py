# -*- coding: utf-8 -*-
"""
Strategy engine: FX Relative-Value + simple hedging overlay.
Public version:
- signal: z-score of returns vs rolling mean/std (mean-reversion)
- position sizing: risk parity style using rolling volatility
- costs: linear bps
- output: equity curve, positions, diagnostics
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def rolling_zscore(x: np.ndarray, window: int = 20) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    z = np.zeros_like(x)
    for t in range(len(x)):
        lo = max(0, t - window + 1)
        w = x[lo:t+1]
        m = np.mean(w)
        s = np.std(w) + 1e-12
        z[t] = (x[t] - m) / s
    return z


def rolling_vol(x: np.ndarray, window: int = 20) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    v = np.zeros_like(x)
    for t in range(len(x)):
        lo = max(0, t - window + 1)
        w = x[lo:t+1]
        v[t] = np.std(w) + 1e-12
    return v


def backtest_mean_reversion(
    r: np.ndarray,
    z_window: int = 20,
    entry_z: float = 1.0,
    exit_z: float = 0.2,
    vol_window: int = 20,
    target_daily_vol: float = 0.01,
    cost_bps: float = 0.5,
) -> dict:
    """
    Simple single-asset MR strategy:
    - if z > entry: short
    - if z < -entry: long
    - exit when |z| < exit
    position scaled by target vol / rolling vol
    costs applied on position changes (bps of notional, approximated)
    """
    r = np.asarray(r, dtype=float)
    z = rolling_zscore(r, window=z_window)
    v = rolling_vol(r, window=vol_window)

    pos = np.zeros_like(r)
    pnl = np.zeros_like(r)

    for t in range(1, len(r)):
        # carry previous by default
        pos[t] = pos[t - 1]

        # entry/exit logic
        if pos[t - 1] == 0:
            if z[t] > entry_z:
                pos[t] = -1.0
            elif z[t] < -entry_z:
                pos[t] = +1.0
        else:
            if abs(z[t]) < exit_z:
                pos[t] = 0.0

        # risk scaling
        scale = float(target_daily_vol / v[t]) if v[t] > 0 else 1.0
        pos[t] *= scale

        # costs: proportional to turnover
        turnover = abs(pos[t] - pos[t - 1])
        cost = (cost_bps * 1e-4) * turnover

        pnl[t] = pos[t - 1] * r[t] - cost

    equity = np.cumsum(pnl)
    return {
        "pnl": pnl,
        "equity": equity,
        "pos": pos,
        "z": z,
        "vol": v,
    }


def to_frame(index: pd.DatetimeIndex, bt: dict) -> pd.DataFrame:
    df = pd.DataFrame({
        "pnl": bt["pnl"],
        "equity": bt["equity"],
        "pos": bt["pos"],
        "z": bt["z"],
        "vol": bt["vol"],
    }, index=index)
    return df
