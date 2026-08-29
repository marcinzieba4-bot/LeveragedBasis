# Derive-native scenarios (presentation backtest, Feb 2024 -> Aug 28 2026)

Both scenarios live fully on Derive under PM2 portfolio margin (collateral and
perp shocked together -> no directional liquidation). Data: Derive funding +
USDC money-market history, Deribit DVOL, Coinbase closes, DeFi Llama staking.

## ETH: 3x loop + 3x options overlay switched on IV>RV

- Loop: weETH -> borrow USDC -> 3x long, short 3x ETH-PERP.
  Loop only: +181% total, +50.0% CAGR, maxDD -1.9%, 1 negative month of 31.
- Overlay: short delta-hedged strangles via RFQ, up to 3x notional, ON only
  when DVOL >= trailing RV30 + 3pts (no lookahead). Cost model: DVOL - 1.5
  vol pts all-in (fair RFQ fill), daily accrual 0.5*(IV^2 - RV^2).
  Switched overlay positive EVERY year 2022-2026 (+0.1..+2.2%/1x notional);
  always-on lost money in 2025 (-4.7%) and 2026 (-1.9%). Signal OFF today
  (DVOL 50.5 vs RV30 68.9).
- Combined: +229% total, +59.5% CAGR, maxDD -3.9%, worst month -3.3%.

## BTC: 6-7x loop (LBTC/cbBTC collateral)

- 6.5x: +383% total, +85.3% CAGR, maxDD -16.8%, 7 negative months of 31
  (funding drought mid-2025: 4 consecutive negative months).
- Capacity: USDC pool $64M supplied / $42.5M borrowed; fitted rate curve
  (1200 obs): <=70% util ~4%, 80-85% ~7.2%, max 17%. Equity ceiling ~ $1M
  at 6.5x before self-inflicted borrow-rate impact.
- Margin: in-account LBTC haircut per API (3.9%) -> fits with headroom;
  docs' conservative reading (~15%) -> max ~3.8x. Confirm with a live test
  position before scaling past 3x.

Scripts: `overlay_signal` logic in this folder's rebacktest lineage; series in
scratch parquets regenerable via `fetch_real.py` + Deribit DVOL endpoint.
