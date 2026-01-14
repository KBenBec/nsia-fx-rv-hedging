# -*- coding: utf-8 -*-
"""
Run: backtest strategy + compute risk (VaR/ES) + stress shocks.
"""

import numpy as np
import pandas as pd

from src.fx_calibration_multistart import load_fx_csv, log_returns_from_prices
from src.fx_strategy_engine import backtest_mean_reversion
from src.fx_risk_var_es import summary_stats


def apply_stress_shock(r: np.ndarray, shock_bps: float = 50.0, prob: float = 0.02, seed: int = 0) -> np.ndarray:
    """
    Stress: occasional adverse shocks in returns (bps in log-return approx).
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(r, dtype=float).copy()
    mask = rng.random(len(r)) < prob
    r[mask] -= shock_bps * 1e-4
    return r


def main():
    s = load_fx_csv("data/sample_fx_rates.csv", price_col="rate")
    px = s.values
    r = log_returns_from_prices(px)

    # base backtest
    bt = backtest_mean_reversion(
        r=r,
        z_window=20,
        entry_z=1.0,
        exit_z=0.2,
        vol_window=20,
        target_daily_vol=0.01,
        cost_bps=0.5,
    )
    stats = summary_stats(bt["pnl"], bt["equity"])
    print("=== BASE strategy stats ===")
    print(stats)

    # stress scenario
    r_stress = apply_stress_shock(r, shock_bps=80.0, prob=0.03, seed=123)
    bt_s = backtest_mean_reversion(r_stress, cost_bps=0.5)
    stats_s = summary_stats(bt_s["pnl"], bt_s["equity"])
    print("\n=== STRESS strategy stats (shock) ===")
    print(stats_s)


if __name__ == "__main__":
    main()
