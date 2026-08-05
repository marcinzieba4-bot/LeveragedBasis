# LeveragedBasis

Backtest of leveraged ETH basis strategy (delta-neutral, 5× loop).

## Website page

`web/logic/` contains the standalone "Logic" page (methodology, simulated
backtest, live PnL section) meant for deployment onto the existing website's
static hosting — see `web/logic/README.md` for deployment steps.
`fetch_derive_pnl.py` publishes the live PnL snapshot the page reads.
