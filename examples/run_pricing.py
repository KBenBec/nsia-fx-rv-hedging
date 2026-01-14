# -*- coding: utf-8 -*-
"""
Run: synthetic FX path generation + returns distribution checks.
"""

import numpy as np
import pandas as pd

from src.fx_regime_svj import SVJParams, simulate_svj_returns


def main():
    rng = np.random.default_rng(42)

    params = SVJParams(
        p11=0.96, p22=0.92,
        sigma1=0.003, sigma2=0.010,
        lam=0.02, mu_j=0.0, sigma_j=0.015
    )

    n = 1000
    r = simulate_svj_returns(n=n, params=params, rng=rng)
    px = 650.0 * np.exp(np.cumsum(r))  # synthetic FX level
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    df = pd.DataFrame({"date": dates, "pair": "EURXOF", "rate": px})
    print(df.head())

    print("\nBasic stats on returns:")
    print("mean:", float(np.mean(r)))
    print("std :", float(np.std(r, ddof=1)))
    print("min :", float(np.min(r)))
    print("max :", float(np.max(r)))

    # Save sample (public/synthetic)
    out = "data/sample_fx_rates.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved synthetic sample to {out}")


if __name__ == "__main__":
    main()
