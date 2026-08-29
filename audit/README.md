# Strategy audit — real-data re-backtest (Jan 2022 → Aug 2026)

Independent audit of `backtest.py` / `fetch_data.py` and the published report.
Full write-up: see the "Leveraged Basis Under Audit" artifact report.

## Headline findings

| | Report claim | Reality |
|---|---|---|
| Funding data | "Binance ETH-USDT" | **Simulated** (hand-tuned AR(1) regimes in `fetch_data.py`) |
| Avg funding APR | +15.4% | **+6.0%** (real Binance archives, Jan 22 – Aug 26) |
| Total return | +487.9% | **+91.3%** with the report's own parameters on real data |
| Liquidation | "80% LTV leaves a buffer" | Buffer is **1.23%**; breached in **16/19 quarters** (USDC debt vs ETH-priced collateral; Aave can't see the perp hedge) |
| Perp margin | not modeled | ~30% of equity must sit on the perp venue |
| Post Mar-2025 | "backtest" | fully invented price/funding/borrow paths |

## Variants re-run on real data

| Variant | Leverage | Total | CAGR | 2026 YTD | Liquidated? |
|---|---|---|---|---|---|
| V0 report replica, real data | 3.36× | +91.3% | +14.9% | +2.9% | not checked (as report) |
| V1 + margin reserve, real yields | 2.35× | +67.3% | +11.7% | +0.8% | **May 2022** |
| V2 survivable loop (4×70%) | 1.77× | +54.2% | +9.7% | +1.4% | fragile |
| V3 Derive weETH + short ETH-PERP | 1.00× | **+66.8%** | **+11.6%** | **+7.6%** | no |

V3 (unlevered, same-venue on Derive) beats the realistic loop with none of the
liquidation risk: Derive ETH-PERP funding averaged **+12.5% APR** since Dec 2023
(vs Binance +6.0%), weETH is accepted at a 7.9% IM haircut, and the margin
engine nets the hedge in one account.

BTC deep-ITM variant: implied financing at mark is 1.2–3.4% APR vs ~12% funding
(≈ +17%/yr at 2×), **but** 0/38 deep-ITM BTC calls had a live ask — RFQ-only;
screen spreads imply 20–340% financing. Pilot via RFQ before trusting it.

## Files

- `fetch_real.py` — downloads all real data (Binance archive dumps, OKX, Deribit,
  Coinbase, DeFi Llama, Derive API). Run with `all` or a source name.
- `rebacktest.py` — the four variants above + liquidation checks.
- `critique_chart.png` — claimed vs real equity curves; funding assumption vs reality.
- `rebt_v1_quarterly.csv`, `rebt_v3_quarterly.csv` — quarterly carry decomposition.
