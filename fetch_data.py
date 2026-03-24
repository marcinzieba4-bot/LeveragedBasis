"""
Data generation for leveraged basis backtest.

No external API calls — all data is synthesised from publicly documented
historical price and rate anchors:

  ETH price:     Key anchor points from CoinGecko / CMC historical records,
                 with log-interpolation + realistic Brownian noise.

  Funding rates: AR(1) model with regime-dependent mean / vol / momentum
                 calibrated to Binance/Bybit published historical averages.

  USDC borrow:   Piecewise-linear model fitted to Aave v2/v3 + Compound
                 quarterly published data and known market events.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone


# ── ETH Price anchors (date, USD) ────────────────────────────────────────────
# Sources: CoinMarketCap / CoinGecko historical data

ETH_PRICE_ANCHORS = [
    ("2020-01-01",  130),
    ("2020-02-15",  270),
    ("2020-03-13",  100),   # COVID crash
    ("2020-05-01",  200),
    ("2020-08-01",  390),   # DeFi summer
    ("2020-09-15",  360),
    ("2020-10-01",  350),
    ("2020-11-01",  440),
    ("2020-12-01",  620),
    ("2021-01-01",  740),
    ("2021-01-25", 1300),
    ("2021-02-05", 1700),
    ("2021-02-20", 1950),
    ("2021-03-13", 1900),
    ("2021-04-22", 2400),
    ("2021-05-12", 4150),   # local ATH
    ("2021-05-24", 1800),   # Elon/China crash
    ("2021-06-26", 1900),
    ("2021-07-20", 1800),
    ("2021-08-01", 2700),
    ("2021-09-07", 3200),
    ("2021-09-20", 2700),
    ("2021-10-20", 4000),
    ("2021-11-10", 4850),   # ATH
    ("2021-12-01", 4600),
    ("2021-12-20", 3750),
    ("2022-01-01", 3700),
    ("2022-01-24", 2200),
    ("2022-02-01", 2700),
    ("2022-03-01", 2900),
    ("2022-04-01", 3400),
    ("2022-04-25", 3000),
    ("2022-05-09", 2400),   # Luna collapse starts
    ("2022-05-18", 1800),
    ("2022-06-13", 1050),
    ("2022-06-18",  900),
    ("2022-07-01", 1100),
    ("2022-07-15", 1200),
    ("2022-08-01", 1700),
    ("2022-09-15", 1600),   # The Merge
    ("2022-10-01", 1350),
    ("2022-11-09", 1200),   # FTX collapse
    ("2022-11-15", 1100),
    ("2022-12-01", 1250),
    ("2022-12-31", 1200),
    ("2023-01-01", 1200),
    ("2023-01-20", 1600),
    ("2023-02-15", 1650),
    ("2023-03-11", 1700),   # SVB / USDC depeg
    ("2023-04-01", 1880),
    ("2023-04-15", 2100),
    ("2023-05-01", 1850),
    ("2023-06-01", 1900),
    ("2023-07-01", 1930),
    ("2023-08-01", 1850),
    ("2023-09-01", 1630),
    ("2023-10-01", 1620),
    ("2023-11-01", 1800),
    ("2023-12-01", 2100),
    ("2023-12-31", 2280),
    ("2024-01-10", 2450),   # BTC ETF approval
    ("2024-01-23", 2270),
    ("2024-02-01", 2250),
    ("2024-02-15", 2700),
    ("2024-03-01", 3400),
    ("2024-03-12", 4050),   # local ATH
    ("2024-04-01", 3500),
    ("2024-04-15", 3000),
    ("2024-05-01", 3050),
    ("2024-05-20", 3700),   # ETH ETF rumours
    ("2024-06-01", 3800),
    ("2024-07-01", 3400),
    ("2024-07-15", 3200),
    ("2024-08-01", 3150),
    ("2024-08-15", 2600),   # summer slump
    ("2024-09-01", 2500),
    ("2024-10-01", 2600),
    ("2024-11-01", 2500),
    ("2024-11-15", 3100),   # post-Trump election
    ("2024-12-01", 3700),
    ("2024-12-31", 3700),
    ("2025-01-01", 3400),
    ("2025-01-20", 3300),
    ("2025-02-01", 2800),
    ("2025-02-15", 2700),
    ("2025-03-01", 2200),
    ("2025-03-24", 2000),
]


def build_eth_price_series(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    anchors = pd.DataFrame(ETH_PRICE_ANCHORS, columns=["date", "price"])
    anchors["date"] = pd.to_datetime(anchors["date"], utc=True)
    anchors = anchors.set_index("date")["price"]

    start = pd.Timestamp("2020-01-01", tz="UTC")
    end   = pd.Timestamp("2025-03-24", tz="UTC")
    all_dates = pd.date_range(start, end, freq="D", tz="UTC")

    # Log-interpolate anchors onto daily grid
    log_anchors = np.log(anchors)
    log_series  = log_anchors.reindex(log_anchors.index.union(all_dates))
    log_series  = log_series.interpolate(method="time")
    log_series  = log_series.reindex(all_dates)

    # Add intraday-style noise that mean-reverts to the anchor path.
    # Use an Ornstein-Uhlenbeck process around the interpolated log price.
    sigma = 0.025   # daily vol ~2.5%
    kappa = 0.20    # mean-reversion speed (strong pull)
    log_base = log_series.values
    log_noisy = np.zeros(len(log_base))
    log_noisy[0] = log_base[0]
    for t in range(1, len(log_base)):
        eps = rng.normal(0, sigma)
        log_noisy[t] = (log_noisy[t-1]
                        + kappa * (log_base[t] - log_noisy[t-1])
                        + eps)

    prices = np.exp(log_noisy)

    df = pd.DataFrame({"timestamp": all_dates, "price": prices})
    return df


# ── Funding rate model ────────────────────────────────────────────────────────

REGIME_PARAMS = {
    # date → {mean daily rate, daily vol, momentum beta}
    # "daily_funding_rate" = sum of 3 × 8-h rates (annualise by ×365)
    "2020-01-01": {"mean": 0.00030, "vol": 0.00030, "mom": 0.0010},
    "2020-08-01": {"mean": 0.00060, "vol": 0.00050, "mom": 0.0015},
    "2020-09-01": {"mean": 0.00080, "vol": 0.00060, "mom": 0.0018},
    "2020-11-01": {"mean": 0.00050, "vol": 0.00040, "mom": 0.0012},
    "2021-01-01": {"mean": 0.00120, "vol": 0.00100, "mom": 0.0025},
    "2021-05-13": {"mean":-0.00020, "vol": 0.00150, "mom":-0.0030},
    "2021-07-01": {"mean": 0.00040, "vol": 0.00060, "mom": 0.0015},
    "2021-10-01": {"mean": 0.00080, "vol": 0.00070, "mom": 0.0020},
    "2021-12-01": {"mean": 0.00020, "vol": 0.00060, "mom": 0.0008},
    "2022-01-01": {"mean": 0.00010, "vol": 0.00070, "mom": 0.0005},
    "2022-04-01": {"mean": 0.00000, "vol": 0.00080, "mom":-0.0005},
    "2022-05-10": {"mean":-0.00100, "vol": 0.00200, "mom":-0.0040},
    "2022-06-01": {"mean":-0.00030, "vol": 0.00100, "mom":-0.0015},
    "2022-07-01": {"mean":-0.00010, "vol": 0.00060, "mom":-0.0008},
    "2022-09-01": {"mean":-0.00010, "vol": 0.00050, "mom":-0.0005},
    "2022-11-01": {"mean":-0.00020, "vol": 0.00080, "mom":-0.0010},
    "2023-01-01": {"mean": 0.00020, "vol": 0.00040, "mom": 0.0006},
    "2023-06-01": {"mean": 0.00030, "vol": 0.00040, "mom": 0.0008},
    "2024-01-01": {"mean": 0.00080, "vol": 0.00080, "mom": 0.0020},
    "2024-03-01": {"mean": 0.00120, "vol": 0.00100, "mom": 0.0025},
    "2024-06-01": {"mean": 0.00050, "vol": 0.00060, "mom": 0.0012},
    "2024-09-01": {"mean": 0.00040, "vol": 0.00050, "mom": 0.0010},
    "2025-01-01": {"mean": 0.00040, "vol": 0.00050, "mom": 0.0010},
}


def build_funding_rate_series(price_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dates  = pd.to_datetime(price_df["timestamp"].values)
    prices = price_df["price"].values
    n      = len(dates)

    regime_keys = sorted(REGIME_PARAMS.keys())

    regime_mean = np.zeros(n)
    regime_vol  = np.zeros(n)
    regime_mom  = np.zeros(n)
    date_strs   = dates.strftime("%Y-%m-%d").tolist()

    for i, ds in enumerate(date_strs):
        best = regime_keys[0]
        for rk in regime_keys:
            if rk <= ds:
                best = rk
        p = REGIME_PARAMS[best]
        regime_mean[i] = p["mean"]
        regime_vol[i]  = p["vol"]
        regime_mom[i]  = p["mom"]

    # 14-day return for momentum signal
    ret14 = np.zeros(n)
    for i in range(14, n):
        ret14[i] = (prices[i] - prices[i - 14]) / prices[i - 14]

    # AR(1) with regime-dependent parameters
    phi = 0.85
    funding = np.zeros(n)
    funding[0] = regime_mean[0]

    for i in range(1, n):
        eps = rng.normal(0, regime_vol[i])
        funding[i] = (phi * funding[i - 1]
                      + (1 - phi) * regime_mean[i]
                      + regime_mom[i] * ret14[i]
                      + eps)
        funding[i] = np.clip(funding[i], -0.0050, 0.0050)

    return pd.DataFrame({"timestamp": price_df["timestamp"].values,
                         "daily_funding_rate": funding})


# ── USDC borrow rates ─────────────────────────────────────────────────────────

USDC_BORROW_ANCHORS = [
    ("2020-01-01", 0.050),
    ("2020-05-01", 0.080),
    ("2020-09-01", 0.200),
    ("2020-11-01", 0.070),
    ("2021-01-01", 0.060),
    ("2021-03-01", 0.120),
    ("2021-05-01", 0.180),
    ("2021-07-01", 0.060),
    ("2021-11-01", 0.080),
    ("2022-01-01", 0.050),
    ("2022-04-01", 0.080),
    ("2022-05-15", 0.250),
    ("2022-06-01", 0.100),
    ("2022-07-01", 0.030),
    ("2022-12-01", 0.020),
    ("2023-03-01", 0.040),
    ("2023-06-01", 0.050),
    ("2023-09-01", 0.060),
    ("2024-01-01", 0.080),
    ("2024-03-01", 0.120),
    ("2024-06-01", 0.070),
    ("2024-09-01", 0.060),
    ("2025-01-01", 0.070),
    ("2025-03-24", 0.060),
]


def build_usdc_borrow_rate_series(date_index: pd.DatetimeIndex) -> pd.Series:
    anchors = pd.DataFrame(USDC_BORROW_ANCHORS, columns=["date", "rate"])
    anchors["date"] = pd.to_datetime(anchors["date"], utc=True)
    anchors = anchors.set_index("date")["rate"]
    full_idx = anchors.index.union(date_index)
    anchors_full = anchors.reindex(full_idx).interpolate(method="time")
    return anchors_full.reindex(date_index)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Building ETH price series …")
    price = build_eth_price_series()
    price.to_parquet("eth_price.parquet", index=False)
    start = pd.to_datetime(price["timestamp"].iloc[0]).date()
    end   = pd.to_datetime(price["timestamp"].iloc[-1]).date()
    print(f"  → {len(price):,} daily rows  ({start} → {end})")
    print(f"     Start price: ${price['price'].iloc[0]:,.0f}  |  "
          f"End price: ${price['price'].iloc[-1]:,.0f}")

    print("Building funding rate series …")
    funding = build_funding_rate_series(price)
    funding.to_parquet("funding_rates.parquet", index=False)
    avg_apr = funding["daily_funding_rate"].mean() * 365 * 100
    pct_pos = (funding["daily_funding_rate"] > 0).mean() * 100
    print(f"  → {len(funding):,} daily rows")
    print(f"     Avg funding APR: {avg_apr:.2f}%  |  {pct_pos:.1f}% of days positive")

    print("Building USDC borrow rate series …")
    usdc_rates = build_usdc_borrow_rate_series(
        pd.DatetimeIndex(price["timestamp"])
    )
    usdc_df = pd.DataFrame({
        "timestamp":       price["timestamp"],
        "usdc_borrow_apr": usdc_rates.values,
    })
    usdc_df.to_parquet("usdc_borrow.parquet", index=False)
    avg_b = usdc_df["usdc_borrow_apr"].mean() * 100
    print(f"  → Avg USDC borrow APR: {avg_b:.2f}%")

    print("\nAll data ready.  Run:  python3 backtest.py")


if __name__ == "__main__":
    main()
