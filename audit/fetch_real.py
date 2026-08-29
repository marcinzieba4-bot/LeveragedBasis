"""Fetch REAL market data, Jan 2022 - Aug 2026, for the leveraged basis re-backtest."""
import requests, time, json, sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone

OUT = "/tmp/claude-0/-home-user-LeveragedBasis/7ea8af11-d669-510b-a966-0b228ec95de9/scratchpad"
START = pd.Timestamp("2022-01-01")
END   = pd.Timestamp("2026-08-28")

S = requests.Session()
S.headers["User-Agent"] = "research/1.0"


def coinbase_daily(product):
    """Daily candles from Coinbase Exchange, paginated 300 at a time."""
    rows = []
    t0 = START
    while t0 <= END:
        t1 = min(t0 + pd.Timedelta(days=299), END + pd.Timedelta(days=1))
        r = S.get(
            f"https://api.exchange.coinbase.com/products/{product}/candles",
            params={"granularity": 86400,
                    "start": t0.strftime("%Y-%m-%dT00:00:00Z"),
                    "end": t1.strftime("%Y-%m-%dT00:00:00Z")},
            timeout=30)
        r.raise_for_status()
        rows += r.json()
        t0 = t1
        time.sleep(0.35)
    df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "vol"])
    df["date"] = pd.to_datetime(df["ts"], unit="s").dt.normalize()
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df[["date", "close"]].rename(columns={"close": "price"})


def deribit_funding(instrument):
    """Deribit hourly funding (interest_8h snapshots); aggregate to daily sum of 1h rates."""
    rows = []
    t0 = int(START.timestamp() * 1000)
    end_ms = int((END + pd.Timedelta(days=1)).timestamp() * 1000)
    while t0 < end_ms:
        t1 = min(t0 + 30 * 86400 * 1000, end_ms)
        r = S.get("https://www.deribit.com/api/v2/public/get_funding_rate_history",
                  params={"instrument_name": instrument,
                          "start_timestamp": t0, "end_timestamp": t1}, timeout=30)
        r.raise_for_status()
        rows += r.json().get("result", [])
        t0 = t1
        time.sleep(0.25)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "daily_funding"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
    # interest_1h is the funding accrued that hour (as a fraction)
    daily = df.groupby("date")["interest_1h"].sum().reset_index()
    daily.columns = ["date", "daily_funding"]
    return daily


def okx_funding(inst):
    """OKX 8h funding history, paginated backwards from now."""
    rows = []
    after = None
    for _ in range(500):
        params = {"instId": inst, "limit": 100}
        if after:
            params["after"] = after
        r = S.get("https://www.okx.com/api/v5/public/funding-rate-history",
                  params=params, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            break
        rows += data
        after = data[-1]["fundingTime"]
        if int(after) < int(START.timestamp() * 1000):
            break
        time.sleep(0.25)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "daily_funding"])
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["date"] = pd.to_datetime(df["fundingTime"].astype(np.int64), unit="ms").dt.normalize()
    daily = df.groupby("date")["fundingRate"].sum().reset_index()
    daily.columns = ["date", "daily_funding"]
    return daily.sort_values("date").reset_index(drop=True)


def derive_funding(instrument):
    """Derive funding history (rate is per-hour), paginated with start/end timestamps."""
    rows = []
    end_ms = int((END + pd.Timedelta(days=1)).timestamp() * 1000)
    t0 = int(START.timestamp() * 1000)
    # walk in 30d windows
    cur = t0
    while cur < end_ms:
        nxt = min(cur + 30 * 86400 * 1000, end_ms)
        r = S.post("https://api.lyra.finance/public/get_funding_rate_history",
                   json={"instrument_name": instrument,
                         "start_timestamp": cur, "end_timestamp": nxt},
                   timeout=30)
        r.raise_for_status()
        res = r.json().get("result", {}).get("funding_rate_history", [])
        rows += res
        cur = nxt
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "daily_funding"])
    df["funding_rate"] = df["funding_rate"].astype(float)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
    df = df.drop_duplicates("timestamp")
    daily = df.groupby("date")["funding_rate"].sum().reset_index()
    daily.columns = ["date", "daily_funding"]
    return daily.sort_values("date").reset_index(drop=True)


def dl_chart(pool, col="apy"):
    r = S.get(f"https://yields.llama.fi/chart/{pool}", timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["data"])
    df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
    return df


def dl_borrow_chart(pool):
    r = S.get(f"https://yields.llama.fi/chartLendBorrow/{pool}", timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["data"])
    df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
    return df


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"

    if what in ("all", "prices"):
        for prod, name in [("ETH-USD", "eth"), ("BTC-USD", "btc")]:
            df = coinbase_daily(prod)
            df.to_parquet(f"{OUT}/real_{name}_price.parquet")
            print(f"{name} price: {len(df)} rows {df['date'].min().date()} -> {df['date'].max().date()}"
                  f"  first={df['price'].iloc[0]:.0f} last={df['price'].iloc[-1]:.0f}")

    if what in ("all", "deribit"):
        for inst, name in [("ETH-PERPETUAL", "eth"), ("BTC-PERPETUAL", "btc")]:
            df = deribit_funding(inst)
            df.to_parquet(f"{OUT}/real_deribit_funding_{name}.parquet")
            ann = df["daily_funding"].mean() * 365 * 100
            print(f"deribit {name} funding: {len(df)} days, avg APR {ann:.2f}%  "
                  f"{df['date'].min().date()} -> {df['date'].max().date()}")

    if what in ("all", "okx"):
        for inst, name in [("ETH-USDT-SWAP", "eth"), ("BTC-USDT-SWAP", "btc")]:
            df = okx_funding(inst)
            df = df[(df["date"] >= START) & (df["date"] <= END)]
            df.to_parquet(f"{OUT}/real_okx_funding_{name}.parquet")
            ann = df["daily_funding"].mean() * 365 * 100
            print(f"okx {name} funding: {len(df)} days, avg APR {ann:.2f}%  "
                  f"{df['date'].min().date()} -> {df['date'].max().date()}")

    if what in ("all", "derive"):
        for inst, name in [("ETH-PERP", "eth"), ("BTC-PERP", "btc")]:
            df = derive_funding(inst)
            df.to_parquet(f"{OUT}/real_derive_funding_{name}.parquet")
            if len(df):
                ann = df["daily_funding"].mean() * 365 * 100
                print(f"derive {name} funding: {len(df)} days, avg APR {ann:.2f}%  "
                      f"{df['date'].min().date()} -> {df['date'].max().date()}")
            else:
                print(f"derive {name} funding: EMPTY")

    if what in ("all", "llama"):
        pools = {
            "aave_usdc": "aa70268e-4b52-42bf-a116-608b370f9501",
            "comp_usdc": "7da72d09-56ca-4ec5-a45f-59114353e487",
            "lido_steth": "747c1d2a-c668-4682-b9f9-296708a3dd90",
        }
        for name, pool in pools.items():
            try:
                df = dl_borrow_chart(pool)
                df.to_parquet(f"{OUT}/real_dl_borrow_{name}.parquet")
                print(f"dl borrow {name}: {len(df)} rows cols={list(df.columns)} "
                      f"{df['date'].min().date()} -> {df['date'].max().date()}")
            except Exception as e:
                print(f"dl borrow {name}: FAILED {e}")
            try:
                df = dl_chart(pool)
                df.to_parquet(f"{OUT}/real_dl_supply_{name}.parquet")
                print(f"dl supply {name}: {len(df)} rows "
                      f"{df['date'].min().date()} -> {df['date'].max().date()}")
            except Exception as e:
                print(f"dl supply {name}: FAILED {e}")
