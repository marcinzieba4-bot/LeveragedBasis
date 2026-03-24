"""
Leveraged Basis Strategy Backtest
===================================
Strategy:
  1. Hold 1 ETH as base collateral.
  2. Loop 5× on a lending protocol (e.g. Aave):
       - Deposit ETH → borrow USDC → buy ETH → repeat
       - At 80 % LTV each loop, 5 loops → ~3.05× leverage
         (geometric series: sum of 0.8^k for k=0..4 ≈ 3.36×;
          or specify exact target leverage below)
  3. Short ETH perpetual futures equal to total long ETH exposure
     → delta-neutral basis trade.
  4. Revenue: 8-h funding rate paid by longs to shorts (Binance ETH-USDT perp).
  5. Cost: USDC borrow interest on borrowed amount.

Output:
  - Daily P&L in USDC
  - Cumulative net yield (%)
  - Funding income vs borrow cost comparison
  - Key statistics (APY, Sharpe, max drawdown, etc.)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ── Strategy parameters ──────────────────────────────────────────────────────

INITIAL_ETH        = 1.0          # starting collateral (ETH)
N_LOOPS            = 5            # number of deposit-borrow-buy loops
LTV_PER_LOOP       = 0.80         # LTV ratio each loop (80%)
PERP_FUNDING_SHARE = 1.0          # fraction of funding we receive (slippage buffer)
BORROW_SPREAD      = 0.005        # extra 0.5 % APR added to raw USDC borrow rate
                                  # (protocol fees, gas amortized, etc.)
FUNDING_TAKE_RATE  = 0.10         # 10% of gross funding goes to protocol/slippage
STAKING_APR        = 0.027        # stETH staking yield (2.7% APR) on all looped ETH

# ── Leverage math ─────────────────────────────────────────────────────────────
# After N loops at LTV each, total ETH held = initial × Σ(LTV^k, k=0..N-1)
leverage_multiplier = sum(LTV_PER_LOOP**k for k in range(N_LOOPS))
# Total ETH long on-chain
# Total USDC borrowed = initial × Σ(LTV^k, k=1..N-1) × price  (in ETH terms)
borrow_eth_multiplier = sum(LTV_PER_LOOP**k for k in range(1, N_LOOPS))


def run_backtest():
    # ── Load data ────────────────────────────────────────────────────────────
    funding = pd.read_parquet("funding_rates.parquet")
    price   = pd.read_parquet("eth_price.parquet")
    usdc    = pd.read_parquet("usdc_borrow.parquet")

    def to_date(col):
        return pd.to_datetime(col).dt.tz_localize(None).dt.normalize()

    funding["date"] = to_date(funding["timestamp"])
    price["date"]   = to_date(price["timestamp"])
    usdc["date"]    = to_date(usdc["timestamp"])

    # funding already has daily_funding_rate column (no aggregation needed)
    daily_funding = funding[["date", "daily_funding_rate"]]

    # Merge everything on date
    df = (
        price[["date", "price"]]
        .merge(daily_funding, on="date", how="left")
        .merge(usdc[["date", "usdc_borrow_apr"]], on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Forward-fill any gaps
    df["daily_funding_rate"] = df["daily_funding_rate"].ffill().fillna(0)
    df["usdc_borrow_apr"]    = df["usdc_borrow_apr"].ffill().fillna(0.05)

    # ── Daily strategy mechanics ─────────────────────────────────────────────
    # We mark-to-market in USD.  ETH position size adjusts with price.
    # Simplification: leverage ratio is held constant (daily rebalance assumed).

    df["eth_long"]       = INITIAL_ETH * leverage_multiplier          # ETH held
    df["eth_short_perp"] = df["eth_long"]                              # delta neutral
    df["usdc_borrowed"]  = INITIAL_ETH * borrow_eth_multiplier * df["price"]  # USD

    # Daily funding income (shorts receive when rate > 0)
    # funding rate × notional of short position
    df["funding_income_usd"] = (
        df["daily_funding_rate"]
        * df["eth_short_perp"]
        * df["price"]
        * (1 - FUNDING_TAKE_RATE)
    )

    # Daily staking income (stETH yield on full leveraged ETH stack)
    # All looped ETH earns staking yield (deposited as stETH collateral)
    df["staking_income_usd"] = (
        STAKING_APR / 365
        * leverage_multiplier
        * INITIAL_ETH
        * df["price"]
    )

    # Daily USDC borrow cost
    df["daily_borrow_rate"]  = (df["usdc_borrow_apr"] + BORROW_SPREAD) / 365
    df["borrow_cost_usd"]    = df["usdc_borrowed"] * df["daily_borrow_rate"]

    # Net daily P&L (funding + staking − borrow)
    df["net_pnl_usd"] = df["funding_income_usd"] + df["staking_income_usd"] - df["borrow_cost_usd"]

    # ── Returns as % of net equity (1 ETH × current price) ─────────────────
    # Net equity = collateral value − borrowed value = 1 ETH × price
    # (the looped ETH minus USDC debt nets back to 1 ETH of equity)
    df["equity_usd"] = INITIAL_ETH * df["price"]

    df["net_yield_daily_pct"]      = df["net_pnl_usd"]        / df["equity_usd"] * 100
    df["funding_yield_daily_pct"]  = df["funding_income_usd"]  / df["equity_usd"] * 100
    df["staking_yield_daily_pct"]  = df["staking_income_usd"]  / df["equity_usd"] * 100
    df["borrow_yield_daily_pct"]   = df["borrow_cost_usd"]     / df["equity_usd"] * 100

    # Annualised view of daily yields (for rolling charts)
    df["net_apr"]     = df["net_yield_daily_pct"]     * 365
    df["funding_apr"] = df["funding_yield_daily_pct"] * 365
    df["staking_apr"] = df["staking_yield_daily_pct"] * 365
    df["borrow_apr"]  = df["borrow_yield_daily_pct"]  * 365

    # Raw funding / borrow APR (for rate-comparison charts)
    df["raw_funding_apr"] = df["daily_funding_rate"] * 365 * 100
    df["raw_borrow_apr"]  = (df["usdc_borrow_apr"] + BORROW_SPREAD) * 100

    # Cumulative USD P&L
    df["cum_net_pnl_usd"]      = df["net_pnl_usd"].cumsum()
    df["cum_funding_usd"]      = df["funding_income_usd"].cumsum()
    df["cum_staking_usd"]      = df["staking_income_usd"].cumsum()
    df["cum_borrow_usd"]       = df["borrow_cost_usd"].cumsum()

    # Cumulative yield-on-equity (compound)
    df["cum_net_yield_pct"]     = df["net_yield_daily_pct"].cumsum()
    df["cum_funding_yield_pct"] = df["funding_yield_daily_pct"].cumsum()
    df["cum_staking_yield_pct"] = df["staking_yield_daily_pct"].cumsum()
    df["cum_borrow_yield_pct"]  = df["borrow_yield_daily_pct"].cumsum()

    return df


def compute_stats(df):
    total_days = len(df)
    years = total_days / 365.25

    total_net_usd      = df["cum_net_pnl_usd"].iloc[-1]
    total_funding_usd  = df["cum_funding_usd"].iloc[-1]
    total_staking_usd  = df["cum_staking_usd"].iloc[-1]
    total_borrow_usd   = df["cum_borrow_usd"].iloc[-1]

    # Annualised yield on equity (simple average of daily net_apr)
    avg_net_apr     = df["net_apr"].mean()
    avg_funding_apr = df["funding_apr"].mean()
    avg_staking_apr = df["staking_apr"].mean()
    avg_borrow_apr  = df["borrow_apr"].mean()

    # Raw rate averages (not leveraged)
    avg_raw_funding = df["raw_funding_apr"].mean()
    avg_raw_borrow  = df["raw_borrow_apr"].mean()
    avg_raw_spread  = avg_raw_funding - avg_raw_borrow

    # Sharpe on daily yield-on-equity
    daily_y = df["net_yield_daily_pct"].values
    sharpe = (daily_y.mean() / daily_y.std()) * np.sqrt(365) if daily_y.std() > 0 else 0

    # Drawdown on cumulative USD P&L
    cum_usd = df["cum_net_pnl_usd"].values
    running_max = np.maximum.accumulate(cum_usd)
    drawdowns = cum_usd - running_max
    max_drawdown_usd = drawdowns.min()

    positive_days = (df["net_pnl_usd"] > 0).sum()
    pct_positive  = positive_days / total_days * 100

    return {
        "total_days":        total_days,
        "years":             years,
        "total_net_usd":     total_net_usd,
        "total_funding_usd": total_funding_usd,
        "total_staking_usd": total_staking_usd,
        "total_borrow_usd":  total_borrow_usd,
        "avg_net_apr":       avg_net_apr,
        "avg_funding_apr":   avg_funding_apr,
        "avg_staking_apr":   avg_staking_apr,
        "avg_borrow_apr":    avg_borrow_apr,
        "avg_raw_funding":   avg_raw_funding,
        "avg_raw_borrow":    avg_raw_borrow,
        "avg_raw_spread":    avg_raw_spread,
        "sharpe":            sharpe,
        "max_drawdown_usd":  max_drawdown_usd,
        "pct_positive_days": pct_positive,
    }


def plot(df, stats):
    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor("#0d1117")
    gs = GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    DARK_BG   = "#0d1117"
    PANEL_BG  = "#161b22"
    GREEN     = "#3fb950"
    RED       = "#f85149"
    BLUE      = "#58a6ff"
    ORANGE    = "#ffa657"
    PURPLE    = "#d2a8ff"
    GREY      = "#8b949e"
    WHITE     = "#e6edf3"

    def style_ax(ax, title=""):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=GREY, labelsize=8)
        ax.xaxis.label.set_color(GREY)
        ax.yaxis.label.set_color(GREY)
        if title:
            ax.set_title(title, color=WHITE, fontsize=10, pad=8, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.grid(axis="y", color="#21262d", linewidth=0.5, linestyle="--")
        ax.grid(axis="x", color="#21262d", linewidth=0.3, linestyle=":")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    dates = df["date"].values

    # ── 1. Cumulative USD P&L breakdown ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(dates, df["cum_funding_usd"], 0, alpha=0.20, color=GREEN)
    ax1.plot(dates, df["cum_funding_usd"],  color=GREEN,  lw=1.5,
             label=f"Cumulative Funding Income  (+${stats['total_funding_usd']:,.0f})")
    ax1.plot(dates, df["cum_staking_usd"],  color=PURPLE, lw=1.5,
             label=f"Cumulative Staking Income  (+${stats['total_staking_usd']:,.0f})")
    ax1.plot(dates, -df["cum_borrow_usd"],  color=RED,    lw=1.5,
             label=f"Cumulative Borrow Cost     (−${stats['total_borrow_usd']:,.0f})")
    ax1.plot(dates, df["cum_net_pnl_usd"],  color=BLUE,   lw=2.5,
             label=f"Net P&L                   (+${stats['total_net_usd']:,.0f}  per 1 ETH collateral)")
    ax1.axhline(0, color=GREY, lw=0.7, linestyle="--")
    ax1.set_ylabel("Cumulative USD P&L (per 1 ETH initial collateral)", color=GREY, fontsize=9)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.15, labelcolor=WHITE)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    style_ax(ax1, f"Cumulative P&L — {N_LOOPS}× ETH Loop (×{leverage_multiplier:.2f} leverage), Delta-Neutral + stETH (2.7%)  |  Jan 2022–Mar 2025")

    # ── 2. Raw funding rate APR on perp ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    roll30_f = df["raw_funding_apr"].rolling(30).mean()
    ax2.fill_between(dates, df["raw_funding_apr"].clip(-100, 250), 0,
                     where=(df["raw_funding_apr"] >= 0), alpha=0.12, color=GREEN)
    ax2.fill_between(dates, df["raw_funding_apr"].clip(-100, 250), 0,
                     where=(df["raw_funding_apr"] < 0),  alpha=0.18, color=RED)
    ax2.plot(dates, df["raw_funding_apr"].clip(-100, 250),
             color=GREY, lw=0.4, alpha=0.4)
    ax2.plot(dates, roll30_f.clip(-100, 250),
             color=GREEN, lw=1.5, label=f"30-day rolling avg")
    ax2.axhline(0, color=GREY, lw=0.7, linestyle="--")
    ax2.axhline(stats["avg_raw_funding"], color=GREEN, lw=1.0, linestyle=":",
                label=f"All-time avg  {stats['avg_raw_funding']:.1f}%")
    ax2.set_ylabel("ETH Perp Funding APR (%)", color=GREY, fontsize=9)
    ax2.legend(loc="upper right", fontsize=7.5, framealpha=0.15, labelcolor=WHITE)
    style_ax(ax2, "ETH Perpetual Funding Rate (Annualised, raw)")

    # ── 3. USDC Borrow APR ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.fill_between(dates, df["raw_borrow_apr"], 0, alpha=0.20, color=RED)
    ax3.plot(dates, df["raw_borrow_apr"], color=RED, lw=1.5,
             label="USDC borrow APR (Aave/Compound + 0.5% spread)")
    ax3.axhline(stats["avg_raw_borrow"], color=ORANGE, lw=1.0, linestyle=":",
                label=f"All-time avg  {stats['avg_raw_borrow']:.1f}%")
    ax3.set_ylabel("USDC Borrow APR (%)", color=GREY, fontsize=9)
    ax3.legend(loc="upper right", fontsize=7.5, framealpha=0.15, labelcolor=WHITE)
    style_ax(ax3, "USDC Borrow Rate on Lending Protocols (Aave / Compound)")

    # ── 4. Net spread on equity ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    net_apr_roll     = df["net_apr"].rolling(30).mean()
    funding_apr_roll = df["funding_apr"].rolling(30).mean()
    ax4.fill_between(dates, df["net_apr"].clip(-200, 600), 0,
                     where=(df["net_apr"] >= 0), alpha=0.18, color=GREEN)
    ax4.fill_between(dates, df["net_apr"].clip(-200, 600), 0,
                     where=(df["net_apr"] < 0),  alpha=0.18, color=RED)
    ax4.plot(dates, funding_apr_roll.clip(-200, 600), color=GREEN, lw=1.0, alpha=0.7,
             label=f"30d Funding APR (avg {stats['avg_funding_apr']:+.1f}%)")
    ax4.plot(dates, net_apr_roll.clip(-200, 600), color=BLUE, lw=1.8,
             label=f"30d Net APR incl. staking (avg {stats['avg_net_apr']:+.1f}%)")
    ax4.axhline(0, color=GREY, lw=0.7, linestyle="--")
    ax4.axhline(stats["avg_staking_apr"], color=PURPLE, lw=1.0, linestyle=":",
                label=f"Staking APR on equity  {stats['avg_staking_apr']:+.1f}%")
    ax4.set_ylabel("APR on Equity (%)", color=GREY, fontsize=9)
    ax4.legend(loc="upper right", fontsize=7.5, framealpha=0.15, labelcolor=WHITE)
    style_ax(ax4, "Net Yield on Equity = Funding + Staking(2.7% × ×3.36) − Borrow")

    # ── 5. ETH Price ──────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.semilogy(dates, df["price"], color=ORANGE, lw=1.5)
    ax5.fill_between(dates, df["price"], df["price"].min(), alpha=0.10, color=ORANGE)
    ax5.set_ylabel("ETH/USD (log scale)", color=GREY, fontsize=9)
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    style_ax(ax5, "ETH/USD Price (synthesised from anchor data)")

    # ── 6. Stats table ────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, :])
    ax6.set_facecolor(PANEL_BG)
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis("off")

    title_str = (
        f"Strategy: {N_LOOPS}× ETH Loop  ·  {LTV_PER_LOOP:.0%} LTV/loop  ·  "
        f"×{leverage_multiplier:.2f} leverage  ·  Delta-Neutral via ETH-USDT Perp Short"
    )
    ax6.text(0.5, 0.97, title_str, transform=ax6.transAxes,
             ha="center", va="top", color=WHITE, fontsize=10, fontweight="bold")

    metrics = [
        ("Period",                        f"Jan 2022 – Mar 2025  ({stats['years']:.1f} yrs)"),
        ("Initial collateral",             f"1 stETH  (~${df['price'].iloc[0]:,.0f})"),
        ("Total Funding Income",           f"+${stats['total_funding_usd']:,.0f} per ETH"),
        ("Total Staking Income (2.7%)",    f"+${stats['total_staking_usd']:,.0f} per ETH"),
        ("Total Borrow Cost",              f"−${stats['total_borrow_usd']:,.0f} per ETH"),
        ("Total Net P&L",                  f"+${stats['total_net_usd']:,.0f} per ETH"),
        ("Avg Net Yield (on equity)",      f"{stats['avg_net_apr']:+.1f}% APR"),
        ("Avg Funding APR on equity",      f"{stats['avg_funding_apr']:+.1f}% APR"),
        ("Avg Staking APR on equity",      f"{stats['avg_staking_apr']:+.1f}% APR"),
        ("Avg Borrow APR on equity",       f"{stats['avg_borrow_apr']:+.1f}% APR"),
        ("Sharpe Ratio",                   f"{stats['sharpe']:.2f}"),
        ("Max Drawdown (USD)",             f"${stats['max_drawdown_usd']:,.0f}"),
        ("% Days Profitable",              f"{stats['pct_positive_days']:.1f}%"),
    ]

    cols = 3
    rows_per_col = (len(metrics) + cols - 1) // cols
    col_w = 1.0 / cols
    row_h = 0.75 / rows_per_col

    for i, (label, value) in enumerate(metrics):
        col = i // rows_per_col
        row = i % rows_per_col
        x = col * col_w + 0.02
        y = 0.85 - row * row_h

        ax6.text(x, y, label + ":", transform=ax6.transAxes,
                 ha="left", va="top", color=GREY, fontsize=8.5)
        is_positive = "+" in value and "−" not in value
        is_negative = "−" in value or ("Borrow" in label and "$" in value)
        color = GREEN if is_positive else (RED if is_negative else WHITE)
        ax6.text(x + col_w * 0.48, y, value, transform=ax6.transAxes,
                 ha="left", va="top", color=color, fontsize=8.5, fontweight="bold")

    ax6.set_title("Summary Statistics", color=WHITE, fontsize=10, pad=8, fontweight="bold")
    for spine in ax6.spines.values():
        spine.set_edgecolor("#30363d")

    fig.suptitle(
        "Leveraged ETH Basis Backtest  |  Jan 2022 – Mar 2025",
        color=WHITE, fontsize=14, fontweight="bold", y=0.995
    )

    plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    print("Chart saved → backtest_results.png")
    plt.close()


def print_quarterly_breakdown(df):
    df = df.copy()
    df["year"]    = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter

    hdr = "─" * 90
    print(f"\n┌{hdr}┐")
    print(f"│{'Quarterly Breakdown — All figures as Annualised % on Equity (per 1 stETH collateral)':^90}│")
    print(f"├{'─'*8}┬{'─'*14}┬{'─'*14}┬{'─'*14}┬{'─'*14}┬{'─'*14}┬{'─'*10}┤")
    print(f"│{'Quarter':^8}│{'Funding APR':^14}│{'Staking APR':^14}│{'Borrow APR':^14}│{'Net APR':^14}│{'Total Income':^14}│{'Days':^10}│")
    print(f"├{'─'*8}┼{'─'*14}┼{'─'*14}┼{'─'*14}┼{'─'*14}┼{'─'*14}┼{'─'*10}┤")

    for (year, q), g in df.groupby(["year", "quarter"]):
        n_days = len(g)
        # Annualise each component: sum of daily % × (365/days_in_quarter)
        ann = 365.0 / n_days
        fund_apr    = g["funding_yield_daily_pct"].sum()  * ann
        stake_apr   = g["staking_yield_daily_pct"].sum()  * ann
        borrow_apr  = g["borrow_yield_daily_pct"].sum()   * ann
        net_apr     = g["net_yield_daily_pct"].sum()      * ann
        total_income_apr = fund_apr + stake_apr

        net_color = "+" if net_apr >= 0 else ""
        print(
            f"│ {year}Q{q}  │"
            f" {fund_apr:+10.2f}%  │"
            f" {stake_apr:+10.2f}%  │"
            f" {borrow_apr:+10.2f}%  │"
            f" {net_apr:+10.2f}%  │"
            f" {total_income_apr:+10.2f}%  │"
            f"  {n_days:>6}    │"
        )

    print(f"├{'─'*8}┼{'─'*14}┼{'─'*14}┼{'─'*14}┼{'─'*14}┼{'─'*14}┼{'─'*10}┤")
    # Full-period averages (annualised)
    ann_all = 365.0 / len(df)
    fund_all    = df["funding_yield_daily_pct"].sum()  * ann_all
    stake_all   = df["staking_yield_daily_pct"].sum()  * ann_all
    borrow_all  = df["borrow_yield_daily_pct"].sum()   * ann_all
    net_all     = df["net_yield_daily_pct"].sum()      * ann_all
    income_all  = fund_all + stake_all
    print(
        f"│{'TOTAL':^8}│"
        f" {fund_all:+10.2f}%  │"
        f" {stake_all:+10.2f}%  │"
        f" {borrow_all:+10.2f}%  │"
        f" {net_all:+10.2f}%  │"
        f" {income_all:+10.2f}%  │"
        f"  {len(df):>6}    │"
    )
    print(f"└{'─'*8}┴{'─'*14}┴{'─'*14}┴{'─'*14}┴{'─'*14}┴{'─'*14}┴{'─'*10}┘")


def print_summary(stats):
    print("\n" + "═" * 68)
    print("  LEVERAGED BASIS STRATEGY — BACKTEST SUMMARY")
    print(f"  {N_LOOPS}× ETH Loop  ·  ×{leverage_multiplier:.2f} leverage  ·  Delta-Neutral via Perp Short")
    print("═" * 68)
    print(f"  Period              : Jan 2022 – Mar 2025 ({stats['years']:.1f} yr)")
    print(f"  Total Funding Income  : +${stats['total_funding_usd']:,.0f} per ETH")
    print(f"  Total Staking Income  : +${stats['total_staking_usd']:,.0f} per ETH  (2.7% × ×{leverage_multiplier:.2f})")
    print(f"  Total Borrow Cost     : −${stats['total_borrow_usd']:,.0f} per ETH")
    print(f"  Total Net P&L         : +${stats['total_net_usd']:,.0f} per ETH")
    print(f"  Avg Net Yield (APR)   : {stats['avg_net_apr']:+.1f}%  on equity")
    print(f"  Avg Funding APR/equity: {stats['avg_funding_apr']:+.1f}%  (leveraged)")
    print(f"  Avg Staking APR/equity: {stats['avg_staking_apr']:+.1f}%  (2.7% × ×{leverage_multiplier:.2f})")
    print(f"  Avg Borrow APR/equity : {stats['avg_borrow_apr']:+.1f}%  (on borrowed/equity)")
    print(f"  Avg raw Funding APR   : {stats['avg_raw_funding']:.1f}%  (raw perp rate)")
    print(f"  Avg raw Borrow APR    : {stats['avg_raw_borrow']:.1f}%  (incl 0.5% spread)")
    print(f"  Sharpe Ratio        : {stats['sharpe']:.2f}")
    print(f"  Max Drawdown (USD)  : ${stats['max_drawdown_usd']:,.0f}")
    print(f"  Days Profitable     : {stats['pct_positive_days']:.1f}%")
    print("═" * 68)


if __name__ == "__main__":
    print("Running leveraged basis backtest …")
    print(f"  Loops: {N_LOOPS}   LTV/loop: {LTV_PER_LOOP:.0%}   Leverage: ×{leverage_multiplier:.2f}")
    print(f"  Borrow spread: +{BORROW_SPREAD*100:.1f}%   Funding take rate: {FUNDING_TAKE_RATE*100:.0f}%\n")

    df = run_backtest()
    stats = compute_stats(df)
    print_summary(stats)
    print_quarterly_breakdown(df)
    plot(df, stats)

    # Save detailed CSV for inspection
    df.to_csv("backtest_detail.csv", index=False)
    print("\nDetailed daily data saved → backtest_detail.csv")
