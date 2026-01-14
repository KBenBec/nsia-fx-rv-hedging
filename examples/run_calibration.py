# -*- coding: utf-8 -*-
"""
Run: load sample FX CSV -> compute log-returns -> calibrate Regime-SVJ (public version).
"""

import numpy as np
import pandas as pd

from src.fx_calibration_multistart import load_fx_csv, log_returns_from_prices, calibrate_from_returns


def main():
    s = load_fx_csv("data/sample_fx_rates.csv", price_col="rate")
    px = s.values
    r = log_returns_from_prices(px)

    # align index (returns length = len(px)-1)
    idx = s.index[1:]

    res = calibrate_from_returns(
        r=r,
        span_vol=20,
        jump_k=4.0,
        multistart_grid=None,
        seed=0,
    )

    print("=== Calibration result ===")
    print("loglik:", res.ll)
    print(res.params)
    print("Regime counts:", {1: int((res.reg == 1).sum()), 2: int((res.reg == 2).sum())})

    # small report
    df = pd.DataFrame({
        "ret": r,
        "regime": res.reg,
    }, index=idx)
    print("\nTail preview:")
    print(df.tail())


if __name__ == "__main__":
    main()
