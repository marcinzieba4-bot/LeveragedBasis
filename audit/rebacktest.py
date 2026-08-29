"""
Leveraged basis strategy re-backtest on REAL data, Jan 2022 - Aug 2026.

Variants:
  V0  Report replica     : original report parameters (5 loops @ 80% LTV, x3.36,
                           flat 2.7% staking, borrow = supply+1%, 90% funding capture),
                           but on REAL funding/price/borrow data. Isolates the effect
                           of the synthetic data on the report's claims.
  V1  Aave loop realistic: same 5x80 loop but with real staking-APR series,
                           borrow = supply*1.23 (utilization-calibrated), perp-margin
                           reserve of 30% of equity, quarterly rebalance costs, and
                           liquidation checks incl. the June-2022 stETH depeg.
  V2  Survivable loop    : 4 loops @ 70% LTV (x2.53), same realism as V1.
  V3  Derive carry       : weETH collateral on Derive + short ETH-PERP, unlevered
                           (0.92x short after IM haircut), Derive funding where it
                           exists (Dec 2023+), Binance funding before, real staking
                           APR, no borrow.
"""
import pandas as pd
import numpy as np

OUT = "/tmp/claude-0/-home-user-LeveragedBasis/7ea8af11-d669-510b-a966-0b228ec95de9/scratchpad"
START, END = pd.Timestamp("2022-01-01"), pd.Timestamp("2026-08-28")


def load_base():
    px = pd.read_parquet(f"{OUT}/real_eth_price.parquet")
    fund = pd.read_parquet(f"{OUT}/real_binance_funding_eth.parquet")
    dfund = pd.read_parquet(f"{OUT}/real_derive_funding_eth.parquet").rename(
        columns={"daily_funding": "derive_funding"})
    steth = pd.read_parquet(f"{OUT}/real_dl_supply_lido_steth.parquet")[["date", "apy"]].rename(
        columns={"apy": "steth_apr"})
    weeth = pd.read_parquet(f"{OUT}/real_dl_supply_weeth.parquet")[["date", "apy"]].rename(
        columns={"apy": "weeth_apr"})
    aave = pd.read_parquet(f"{OUT}/real_dl_supply_aave_usdc.parquet")[["date", "apyBase"]].rename(
        columns={"apyBase": "aave_supply"})
    comp = pd.read_parquet(f"{OUT}/real_dl_supply_comp_usdc.parquet")[["date", "apyBase"]].rename(
        columns={"apyBase": "comp_supply"})

    df = px.merge(fund, on="date", how="left") \
           .merge(dfund, on="date", how="left") \
           .merge(steth, on="date", how="left") \
           .merge(weeth, on="date", how="left") \
           .merge(aave, on="date", how="left") \
           .merge(comp, on="date", how="left")
    df = df[(df["date"] >= START) & (df["date"] <= END)].sort_values("date").reset_index(drop=True)

    df["daily_funding"] = df["daily_funding"].fillna(0)

    # stETH APR: DL series starts May 2022; before that use documented ~4.6%
    df["steth_apr"] = df["steth_apr"].bfill().ffill()

    # weETH APR: exists from Jun 2024; before that proxy with stETH + 0 restaking premium
    df["weeth_apr"] = df["weeth_apr"].fillna(df["steth_apr"])

    # USDC borrow series:
    #   2022-01..2022-09 : report's own Aave-v2 anchors (only non-live stretch; flagged)
    #   2022-10..2023-02 : Compound v3 supply * 1.23
    #   2023-02..present : Aave v3 supply * 1.23  (ratio calibrated on live utilization)
    anchors = [("2022-01-01", 3.2), ("2022-02-01", 3.0), ("2022-03-01", 2.8),
               ("2022-04-01", 3.0), ("2022-05-08", 3.5), ("2022-05-11", 9.0),
               ("2022-05-18", 6.0), ("2022-05-25", 4.0), ("2022-06-01", 3.2),
               ("2022-06-13", 4.5), ("2022-06-20", 3.0), ("2022-07-01", 2.5),
               ("2022-08-01", 2.3), ("2022-09-01", 2.3), ("2022-09-15", 2.5),
               ("2022-10-01", 2.9)]
    anc = pd.DataFrame(anchors, columns=["date", "v2"])
    anc["date"] = pd.to_datetime(anc["date"])
    df = df.merge(anc, on="date", how="left")
    df["v2"] = df["v2"].interpolate("linear")

    df["borrow_apr"] = np.where(
        df["aave_supply"].notna(), df["aave_supply"] * 1.23,
        np.where(df["comp_supply"].notna(), df["comp_supply"] * 1.23, df["v2"]))
    df["borrow_apr"] = pd.Series(df["borrow_apr"]).ffill().clip(lower=0.5) / 100.0

    df["steth_apr"] = df["steth_apr"] / 100.0
    df["weeth_apr"] = df["weeth_apr"] / 100.0
    return df


# stETH/ETH ratio episode: June 2022 depeg (documented lows ~0.9350 mid-June,
# recovering by September). Linear-ish interpolation between documented marks.
STETH_RATIO_MARKS = [
    ("2022-01-01", 1.000), ("2022-05-01", 0.995), ("2022-05-12", 0.975),
    ("2022-06-01", 0.970), ("2022-06-11", 0.945), ("2022-06-17", 0.935),
    ("2022-07-01", 0.955), ("2022-08-01", 0.965), ("2022-09-15", 0.985),
    ("2022-11-01", 0.990), ("2023-01-01", 0.995), ("2023-04-15", 0.999),
    ("2026-08-28", 0.999),
]


def steth_ratio_series(dates):
    m = pd.DataFrame(STETH_RATIO_MARKS, columns=["date", "ratio"])
    m["date"] = pd.to_datetime(m["date"])
    s = m.set_index("date")["ratio"]
    idx = s.index.union(dates)
    return s.reindex(idx).interpolate("time").reindex(dates).values


def run_variant(df, name, n_loops, ltv, margin_reserve, funding_col,
                staking_col, use_borrow, funding_capture, staking_flat=None,
                liq_threshold=0.81, rebalance_cost_q=0.0005, check_liq=True):
    """Carry accounting per 1 unit of equity (USD), quarterly-rebalanced.

    margin_reserve: fraction of equity parked as perp margin (earns nothing).
    Loop runs on (1 - margin_reserve) of equity.
    """
    lev = sum(ltv**k for k in range(n_loops))            # long exposure mult on looped equity
    bor = sum(ltv**k for k in range(1, n_loops))          # borrowed mult on looped equity
    w = 1.0 - margin_reserve
    long_mult = w * lev                                   # ETH long / equity
    borrow_mult = w * bor                                 # USDC debt / equity
    short_mult = long_mult                                # delta neutral

    dates = df["date"]
    ratio = steth_ratio_series(pd.DatetimeIndex(dates))

    fund = df[funding_col].fillna(0).values
    borrow = df["borrow_apr"].values
    stk = (np.full(len(df), staking_flat) if staking_flat is not None
           else df[staking_col].values)

    daily = (funding_capture * fund * short_mult
             + stk / 365 * long_mult
             - (borrow + 0.005) / 365 * borrow_mult * use_borrow)

    # quarterly rebalance cost on total notional turnover (~10 bps on adjustments)
    q_end = dates.dt.to_period("Q").ne(dates.dt.to_period("Q").shift(-1))
    daily = daily - np.where(q_end.values, rebalance_cost_q * (long_mult + short_mult) / 2, 0)

    equity = (1 + pd.Series(daily)).cumprod()

    # Liquidation check: Aave loop LTV_effective = debt / (collateral * stETH_ratio)
    # loop entered at `ltv`; collateral marked at ratio; debt in USD fixed between rebalances.
    liq_date = None
    if check_liq and use_borrow:
        eff_ltv = ltv / ratio          # ratio<1 raises effective LTV
        breach = eff_ltv > liq_threshold
        if breach.any():
            liq_date = dates.iloc[int(np.argmax(breach))]

    res = pd.DataFrame({
        "date": dates, "daily_ret": daily, "equity": equity.values,
        "funding_pnl": funding_capture * fund * short_mult,
        "staking_pnl": stk / 365 * long_mult,
        "borrow_cost": (borrow + 0.005) / 365 * borrow_mult * use_borrow,
    })
    return {"name": name, "df": res, "leverage": long_mult, "borrow_mult": borrow_mult,
            "liq_date": liq_date}


def stats(res):
    r = res["df"]["daily_ret"].values
    eq = res["df"]["equity"].values
    n = len(r)
    total = eq[-1] - 1
    cagr = eq[-1] ** (365.25 / n) - 1
    vol = r.std() * np.sqrt(365)
    sharpe = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0
    run_max = np.maximum.accumulate(eq)
    dd = ((eq - run_max) / run_max).min()
    win = (r > 0).mean() * 100
    return dict(total=total*100, cagr=cagr*100, vol=vol*100, sharpe=sharpe,
                max_dd=dd*100, win=win)


def quarterly(res):
    d = res["df"].copy()
    d["q"] = d["date"].dt.to_period("Q")
    rows = []
    for q, g in d.groupby("q"):
        ann = 365 / len(g)
        rows.append({
            "q": str(q),
            "net_apr": g["daily_ret"].sum() * ann * 100,
            "funding_apr": g["funding_pnl"].sum() * ann * 100,
            "staking_apr": g["staking_pnl"].sum() * ann * 100,
            "borrow_apr": -g["borrow_cost"].sum() * ann * 100,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_base()
    print(f"data: {len(df)} days  {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"avg funding APR (Binance real): {df['daily_funding'].mean()*365*100:+.2f}%")
    print(f"avg borrow APR: {df['borrow_apr'].mean()*100:.2f}%  "
          f"avg stETH APR: {df['steth_apr'].mean()*100:.2f}%")

    # Derive funding series: Binance pre-launch, Derive after
    df["derive_mix"] = df["derive_funding"].where(df["derive_funding"].notna(), df["daily_funding"])

    variants = [
        run_variant(df, "V0 report-replica (5x80, x3.36, flat 2.7%, no margin reserve)",
                    5, 0.80, 0.0, "daily_funding", None, 1.0, 0.90,
                    staking_flat=0.027, rebalance_cost_q=0.0, check_liq=False),
        run_variant(df, "V1 realistic Aave loop (5x80 + 30% perp margin reserve)",
                    5, 0.80, 0.30, "daily_funding", "steth_apr", 1.0, 0.90),
        run_variant(df, "V2 survivable loop (4x70 + 30% reserve)",
                    4, 0.70, 0.30, "daily_funding", "steth_apr", 1.0, 0.90),
        run_variant(df, "V3 Derive carry (weETH + short ETH-PERP, unlevered)",
                    1, 0.0, 0.0, "derive_mix", "weeth_apr", 0.0, 0.90,
                    rebalance_cost_q=0.0002, check_liq=False),
    ]
    # V3 short size = 0.92x (weETH IM haircut) -> scale funding
    # crude: rerun with capture 0.9*0.92
    variants[3] = run_variant(df, "V3 Derive carry (weETH + short ETH-PERP, unlevered)",
                              1, 0.0, 0.0, "derive_mix", "weeth_apr", 0.0, 0.90 * 0.92,
                              rebalance_cost_q=0.0002, check_liq=False)

    print("\n" + "=" * 110)
    print(f"{'variant':<58}{'lev':>5}{'total%':>9}{'CAGR%':>8}{'vol%':>7}"
          f"{'Sharpe':>8}{'maxDD%':>8}{'liq?':>12}")
    print("-" * 110)
    for v in variants:
        s = stats(v)
        liq = str(v["liq_date"].date()) if v["liq_date"] is not None else "no"
        print(f"{v['name']:<58}{v['leverage']:>5.2f}{s['total']:>9.1f}{s['cagr']:>8.1f}"
              f"{s['vol']:>7.1f}{s['sharpe']:>8.2f}{s['max_dd']:>8.1f}{liq:>12}")

    print("\nPer-year net APR by variant:")
    for v in variants:
        d = v["df"].copy()
        d["y"] = d["date"].dt.year
        yr = d.groupby("y")["daily_ret"].mean() * 365 * 100
        print(f"  {v['name'][:44]:<46}" + "  ".join(f"{y}:{val:+6.1f}%" for y, val in yr.items()))

    # save
    for i, v in enumerate(variants):
        v["df"].to_parquet(f"{OUT}/rebt_v{i}.parquet")
    quarterly(variants[1]).to_csv(f"{OUT}/rebt_v1_quarterly.csv", index=False)
    quarterly(variants[3]).to_csv(f"{OUT}/rebt_v3_quarterly.csv", index=False)
    print("\nsaved rebt_v*.parquet")
