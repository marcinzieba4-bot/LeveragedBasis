"""
Data fetching for leveraged basis backtest (Jan 2022 – Mar 2025).

USDC borrow rate sources (in priority order):
  1. Aave v3 USDC/Ethereum (DeFi Llama):  Feb 2023 → Mar 2025
       supply APY fetched directly; borrow APY derived via utilization:
       borrow = supply / (utilization × (1 − reserve_factor))
       utilization = 1 − (tvlUsd / totalSupplyUsd)
       At current 75.3% utilization, borrow/supply ≈ 1.477×.
       We use the daily tvlUsd to approximate utilization over history.
  2. Compound v3 USDC/Ethereum (DeFi Llama): Oct 2022 → Jan 2023
       Same supply→borrow conversion (reserve factor 5%, same utilization method).
  3. Aave v2 model: Jan 2022 → Sep 2022
       Bear-market anchors calibrated from published Aave governance data and
       known market events; much lower than previous model.

ETH price: Ornstein-Uhlenbeck anchored to ~70 documented price levels.
Funding rates: AR(1) regime model calibrated to Binance/Bybit historical averages.
"""

import requests
import pandas as pd
import numpy as np
import time


# ── Constants ─────────────────────────────────────────────────────────────────

START = "2022-01-01"
END   = "2025-03-24"

# Aave v3 / Compound v3 reserve factors
AAVE_V3_RF  = 0.10   # 10%
COMP_V3_RF  = 0.05   # 5%

# ── DeFi Llama helper ─────────────────────────────────────────────────────────

DL_BASE = "https://yields.llama.fi"

AAVE_V3_USDC_POOL = "aa70268e-4b52-42bf-a116-608b370f9501"   # ETH mainnet
COMP_V3_USDC_POOL = "7da72d09-56ca-4ec5-a45f-59114353e487"   # ETH mainnet


def fetch_dl_chart(pool_id: str) -> pd.DataFrame:
    r = requests.get(f"{DL_BASE}/chart/{pool_id}", timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["data"])
    df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
    df = df[["date", "apyBase", "tvlUsd"]].sort_values("date").reset_index(drop=True)
    return df


def fetch_dl_lend_borrow(pool_id: str) -> dict:
    """Fetch current lending/borrowing data to calibrate the conversion factor."""
    r = requests.get(f"{DL_BASE}/lendBorrow", timeout=30)
    r.raise_for_status()
    data = r.json()
    match = [d for d in data if d.get("pool") == pool_id]
    return match[0] if match else {}


def supply_to_borrow(supply_apy: pd.Series,
                     tvl_usd: pd.Series,
                     total_supply_usd: float,
                     reserve_factor: float) -> pd.Series:
    """
    Convert supply APY to borrow APY using utilization derived from daily TVL.

    tvlUsd (DeFi Llama) = free liquidity = totalSupply - totalBorrows
    utilization = 1 - tvlUsd / totalSupplyUsd

    We assume totalSupplyUsd scales proportionally to the most recent snapshot.
    The relationship is:
        supply_apy = borrow_apy × utilization × (1 - reserve_factor)
        borrow_apy = supply_apy / (utilization × (1 - rf))
    """
    # Estimate total supply at each point by scaling from latest known value
    # (tvlUsd changes daily; assume totalSupply = tvlUsd × scale_factor)
    # Current snapshot gives us the scale factor once.
    tvl_latest  = tvl_usd.iloc[-1]
    scale       = total_supply_usd / tvl_latest if tvl_latest > 0 else 1.0
    total_supply_series = tvl_usd * scale
    utilization = 1.0 - tvl_usd / total_supply_series.clip(lower=1e-9)
    utilization = utilization.clip(lower=0.30, upper=0.97)  # sensible bounds

    borrow = supply_apy / (utilization * (1 - reserve_factor))
    return borrow.clip(lower=0.5, upper=80.0)   # hard caps: 0.5% floor, 80% ceiling


# ── USDC borrow rate ──────────────────────────────────────────────────────────

# Aave v2 calibrated anchors: Jan 2022 → Sep 2022
# Sources: Aave governance forums, risk dashboards, published utilization snapshots
# Note: bear market = low utilization = historically low rates.
AAVE_V2_ANCHORS_2022 = [
    # (date,  borrow_APR)
    ("2022-01-01", 3.8),
    ("2022-02-01", 3.5),
    ("2022-03-01", 3.5),
    ("2022-04-01", 3.8),
    ("2022-04-20", 4.2),
    ("2022-05-08", 4.8),
    ("2022-05-11", 11.5),  # Luna collapse starts — utilization spike
    ("2022-05-18",  8.0),
    ("2022-05-25",  5.5),
    ("2022-06-01",  4.0),
    ("2022-06-13",  5.5),  # brief second-leg spike (3AC / Celsius contagion)
    ("2022-06-20",  3.8),
    ("2022-07-01",  2.8),
    ("2022-08-01",  2.5),
    ("2022-09-01",  2.5),
    ("2022-09-15",  2.8),  # Merge
    ("2022-10-01",  2.9),  # Compound v3 launches; data takes over here
]


def build_usdc_borrow_series(all_dates: pd.DatetimeIndex) -> pd.Series:
    """
    Merge three data sources into a single daily USDC borrow APR series.
    """
    # ── 1. Aave v3 (Feb 2023 – present) ──────────────────────────────────────
    print("  Fetching Aave v3 USDC supply rate from DeFi Llama …")
    aave3 = fetch_dl_chart(AAVE_V3_USDC_POOL)
    lb3   = fetch_dl_lend_borrow(AAVE_V3_USDC_POOL)
    total_supply_aave3 = lb3.get("totalSupplyUsd", aave3["tvlUsd"].iloc[-1] / 0.247)
    aave3["borrow_apr"] = supply_to_borrow(
        aave3["apyBase"], aave3["tvlUsd"],
        total_supply_aave3, AAVE_V3_RF
    )
    print(f"    Aave v3: {len(aave3)} pts  "
          f"({aave3['date'].iloc[0].date()} → {aave3['date'].iloc[-1].date()})  "
          f"avg borrow {aave3['borrow_apr'].mean():.2f}%")

    # ── 2. Compound v3 (Oct 2022 – Jan 2023 gap-filler) ──────────────────────
    print("  Fetching Compound v3 USDC supply rate from DeFi Llama …")
    comp3 = fetch_dl_chart(COMP_V3_USDC_POOL)
    lb_c3 = fetch_dl_lend_borrow(COMP_V3_USDC_POOL)
    total_supply_comp3 = lb_c3.get("totalSupplyUsd", comp3["tvlUsd"].iloc[-1] / 0.25)
    comp3["borrow_apr"] = supply_to_borrow(
        comp3["apyBase"], comp3["tvlUsd"],
        total_supply_comp3, COMP_V3_RF
    )
    # Use Compound v3 only where Aave v3 has no data
    aave3_start = aave3["date"].iloc[0]
    comp3_gap = comp3[comp3["date"] < aave3_start].copy()
    print(f"    Compound v3 gap-fill: {len(comp3_gap)} pts  "
          f"({comp3_gap['date'].iloc[0].date() if len(comp3_gap) else 'N/A'} → "
          f"{comp3_gap['date'].iloc[-1].date() if len(comp3_gap) else 'N/A'})")

    # ── 3. Aave v2 anchors (Jan 2022 – Oct 2022) ────────────────────────────
    v2 = pd.DataFrame(AAVE_V2_ANCHORS_2022, columns=["date", "borrow_apr"])
    v2["date"] = pd.to_datetime(v2["date"])

    # ── Merge: v2 anchors → interpolate → fill Compound v3 gap → Aave v3 ──
    # Build the combined daily series
    combined = pd.DataFrame({"date": all_dates})

    # Aave v2 section: daily dates Jan 2022 – Sep 2022
    v2_idx   = pd.date_range("2022-01-01", "2022-10-01", freq="D")
    v2_series = v2.set_index("date")["borrow_apr"].reindex(v2.index.union(v2_idx))
    v2_series.index = pd.to_datetime(v2_series.index)   # ensure DatetimeIndex
    v2_series = v2_series.sort_index().interpolate("time")
    v2_full   = v2_series.reindex(v2_idx).reset_index().rename(columns={"index": "date", 0: "borrow_apr"})
    v2_full.columns = ["date", "borrow_apr"]

    # Stack all rate sources
    all_rates = pd.concat([
        v2_full.rename(columns={"borrow_apr": "borrow_apr"}),
        comp3_gap[["date", "borrow_apr"]],
        aave3[["date", "borrow_apr"]],
    ], ignore_index=True)
    all_rates = all_rates.sort_values("date").drop_duplicates("date", keep="last")

    # Merge onto full date range and interpolate any remaining gaps
    combined = combined.merge(all_rates, on="date", how="left")
    combined["borrow_apr"] = combined["borrow_apr"].interpolate("linear").ffill()
    borrow = combined.set_index("date")["borrow_apr"]

    # Smooth single-day outliers: if a day is >2× the 7-day rolling median,
    # cap it at 2× the rolling median (handles brief DeFi liquidity panics
    # that create data noise without removing genuine bull-market elevation).
    roll7_med = borrow.rolling(7, center=True, min_periods=1).median()
    borrow = borrow.clip(upper=roll7_med * 2.0).clip(upper=40.0, lower=0.5)

    print(f"  Merged USDC borrow APR: mean={borrow.mean():.2f}%  "
          f"max={borrow.max():.2f}%  median={borrow.median():.2f}%  min={borrow.min():.2f}%")
    return borrow


# ── ETH price ─────────────────────────────────────────────────────────────────

ETH_PRICE_ANCHORS = [
    ("2022-01-01", 3700),
    ("2022-01-24", 2200),
    ("2022-02-01", 2700),
    ("2022-03-01", 2900),
    ("2022-04-01", 3400),
    ("2022-04-25", 3000),
    ("2022-05-09", 2400),
    ("2022-05-18", 1800),
    ("2022-06-13", 1050),
    ("2022-06-18",  900),
    ("2022-07-01", 1100),
    ("2022-07-15", 1200),
    ("2022-08-01", 1700),
    ("2022-09-15", 1600),
    ("2022-10-01", 1350),
    ("2022-11-09", 1200),
    ("2022-11-15", 1100),
    ("2022-12-01", 1250),
    ("2022-12-31", 1200),
    ("2023-01-20", 1600),
    ("2023-02-15", 1650),
    ("2023-03-11", 1700),
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
    ("2024-01-10", 2450),
    ("2024-01-23", 2270),
    ("2024-02-15", 2700),
    ("2024-03-12", 4050),
    ("2024-04-01", 3500),
    ("2024-04-15", 3000),
    ("2024-05-20", 3700),
    ("2024-06-01", 3800),
    ("2024-07-01", 3400),
    ("2024-08-15", 2600),
    ("2024-09-01", 2500),
    ("2024-11-01", 2500),
    ("2024-11-15", 3100),
    ("2024-12-01", 3700),
    ("2024-12-31", 3700),
    ("2025-01-20", 3300),
    ("2025-02-15", 2700),
    ("2025-03-01", 2200),
    ("2025-03-24", 2000),
]


def build_eth_price_series(all_dates: pd.DatetimeIndex, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    anchors = pd.DataFrame(ETH_PRICE_ANCHORS, columns=["date", "price"])
    anchors["date"] = pd.to_datetime(anchors["date"])
    anchors = anchors.set_index("date")["price"]

    log_anchors = np.log(anchors)
    full_idx = log_anchors.index.union(all_dates)
    log_series = log_anchors.reindex(full_idx).interpolate("time").reindex(all_dates)

    sigma = 0.025
    kappa = 0.20
    log_base = log_series.values
    log_noisy = np.zeros(len(log_base))
    log_noisy[0] = log_base[0]
    for t in range(1, len(log_base)):
        eps = rng.normal(0, sigma)
        log_noisy[t] = log_noisy[t-1] + kappa * (log_base[t] - log_noisy[t-1]) + eps

    return pd.DataFrame({"timestamp": all_dates, "price": np.exp(log_noisy)})


# ── Funding rate model ────────────────────────────────────────────────────────

REGIME_PARAMS = {
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
    date_strs   = dates.strftime("%Y-%m-%d").tolist()

    regime_mean = np.zeros(n)
    regime_vol  = np.zeros(n)
    regime_mom  = np.zeros(n)
    for i, ds in enumerate(date_strs):
        best = regime_keys[0]
        for rk in regime_keys:
            if rk <= ds:
                best = rk
        p = REGIME_PARAMS[best]
        regime_mean[i] = p["mean"]
        regime_vol[i]  = p["vol"]
        regime_mom[i]  = p["mom"]

    ret14 = np.zeros(n)
    for i in range(14, n):
        ret14[i] = (prices[i] - prices[i - 14]) / prices[i - 14]

    phi = 0.85
    funding = np.zeros(n)
    funding[0] = regime_mean[0]
    for i in range(1, n):
        eps = rng.normal(0, regime_vol[i])
        funding[i] = (phi * funding[i - 1]
                      + (1 - phi) * regime_mean[i]
                      + regime_mom[i] * ret14[i]
                      + eps)
        funding[i] = np.clip(funding[i], -0.005, 0.005)

    return pd.DataFrame({"timestamp": price_df["timestamp"].values,
                         "daily_funding_rate": funding})


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_dates = pd.date_range(START, END, freq="D")
    all_dates_naive = all_dates.tz_localize(None)

    print("Building ETH price series …")
    price = build_eth_price_series(all_dates_naive)
    price.to_parquet("eth_price.parquet", index=False)
    print(f"  {len(price):,} rows  "
          f"${price['price'].iloc[0]:,.0f} → ${price['price'].iloc[-1]:,.0f}")

    print("\nBuilding funding rate series …")
    funding = build_funding_rate_series(price)
    funding.to_parquet("funding_rates.parquet", index=False)
    avg_apr = funding["daily_funding_rate"].mean() * 365 * 100
    pct_pos = (funding["daily_funding_rate"] > 0).mean() * 100
    print(f"  {len(funding):,} rows  avg APR: {avg_apr:.2f}%  {pct_pos:.1f}% positive days")

    print("\nBuilding USDC borrow rate series …")
    borrow = build_usdc_borrow_series(all_dates_naive)
    usdc_df = pd.DataFrame({"timestamp": all_dates_naive, "usdc_borrow_apr": borrow.values / 100})
    usdc_df.to_parquet("usdc_borrow.parquet", index=False)
    print(f"  Avg borrow APR (as decimal): {usdc_df['usdc_borrow_apr'].mean()*100:.2f}%")

    print("\nAll data ready. Run:  python3 backtest.py")


if __name__ == "__main__":
    main()
