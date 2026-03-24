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
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy import stats as scipy_stats
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


def compute_risk_stats(df):
    """Full risk analytics on the daily net yield (% on equity)."""
    r = df["net_yield_daily_pct"].values / 100.0   # daily fraction

    # ── Return metrics ──────────────────────────────────────────────────────
    n_days   = len(r)
    total_ret = (1 + r).prod() - 1                  # compound total return
    cagr      = (1 + total_ret) ** (365.25 / n_days) - 1

    # ── Volatility ──────────────────────────────────────────────────────────
    ann_vol   = r.std() * np.sqrt(365)

    # ── Sharpe (risk-free = 0, carry is the alpha) ───────────────────────────
    sharpe    = (r.mean() / r.std()) * np.sqrt(365) if r.std() > 0 else 0

    # ── Sortino (downside deviation, MAR = 0) ───────────────────────────────
    neg = r[r < 0]
    down_vol  = neg.std() * np.sqrt(365) if len(neg) > 1 else np.nan
    sortino   = (r.mean() * 365) / down_vol if (down_vol and down_vol > 0) else np.nan

    # ── Drawdown series ──────────────────────────────────────────────────────
    equity_curve = (1 + r).cumprod()
    running_max  = np.maximum.accumulate(equity_curve)
    dd_series    = (equity_curve - running_max) / running_max   # negative

    max_dd       = dd_series.min()           # most negative = worst drawdown

    # Drawdown duration: longest consecutive stretch below high-water mark
    in_dd        = dd_series < 0
    max_dur = cur_dur = 0
    for flag in in_dd:
        if flag:
            cur_dur += 1
            max_dur  = max(max_dur, cur_dur)
        else:
            cur_dur = 0

    # Calmar = CAGR / |max drawdown|
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    # ── VaR / CVaR ───────────────────────────────────────────────────────────
    var_95  = np.percentile(r, 5)
    var_99  = np.percentile(r, 1)
    cvar_95 = r[r <= var_95].mean()
    cvar_99 = r[r <= var_99].mean()

    # ── Distribution ─────────────────────────────────────────────────────────
    skew  = scipy_stats.skew(r)
    kurt  = scipy_stats.kurtosis(r)        # excess kurtosis (normal = 0)

    # ── Tail / win stats ─────────────────────────────────────────────────────
    win_rate   = (r > 0).mean() * 100
    avg_win    = r[r > 0].mean() * 100 if (r > 0).any() else 0
    avg_loss   = r[r < 0].mean() * 100 if (r < 0).any() else 0
    pfactor    = (-r[r > 0].sum() / r[r < 0].sum()) if (r < 0).any() else np.nan
    best_day   = r.max() * 100
    worst_day  = r.min() * 100

    # Monthly returns
    monthly = (
        df.set_index("date")["net_yield_daily_pct"]
        .resample("ME").sum()     # sum of daily % ≈ monthly total
    )
    best_month  = monthly.max()
    worst_month = monthly.min()

    return {
        "n_days":       n_days,
        "total_ret":    total_ret * 100,
        "cagr":         cagr * 100,
        "ann_vol":      ann_vol * 100,
        "sharpe":       sharpe,
        "sortino":      sortino,
        "calmar":       calmar,
        "max_dd":       max_dd * 100,
        "max_dd_dur":   max_dur,
        "var_95":       var_95 * 100,
        "var_99":       var_99 * 100,
        "cvar_95":      cvar_95 * 100,
        "cvar_99":      cvar_99 * 100,
        "skew":         skew,
        "kurt":         kurt,
        "win_rate":     win_rate,
        "avg_win":      avg_win,
        "avg_loss":     avg_loss,
        "pfactor":      pfactor,
        "best_day":     best_day,
        "worst_day":    worst_day,
        "best_month":   best_month,
        "worst_month":  worst_month,
        "equity_curve": equity_curve,
        "dd_series":    dd_series,
        "monthly":      monthly,
    }


def print_risk_stats(rs):
    W = 68
    print("\n" + "═" * W)
    print("  RISK STATISTICS")
    print("═" * W)
    rows = [
        ("RETURN",           None),
        ("Total Return",     f"{rs['total_ret']:+.1f}%"),
        ("CAGR",             f"{rs['cagr']:+.1f}%"),
        ("Ann. Volatility",  f"{rs['ann_vol']:.1f}%"),
        ("",                 None),
        ("RISK-ADJUSTED",    None),
        ("Sharpe Ratio",     f"{rs['sharpe']:.2f}"),
        ("Sortino Ratio",    f"{rs['sortino']:.2f}"),
        ("Calmar Ratio",     f"{rs['calmar']:.2f}"),
        ("",                 None),
        ("DRAWDOWN",         None),
        ("Max Drawdown",     f"{rs['max_dd']:.1f}%"),
        ("Max DD Duration",  f"{rs['max_dd_dur']} days"),
        ("",                 None),
        ("TAIL RISK (daily)",None),
        ("VaR 95%",          f"{rs['var_95']:.3f}%"),
        ("VaR 99%",          f"{rs['var_99']:.3f}%"),
        ("CVaR 95%",         f"{rs['cvar_95']:.3f}%"),
        ("CVaR 99%",         f"{rs['cvar_99']:.3f}%"),
        ("",                 None),
        ("DISTRIBUTION",     None),
        ("Skewness",         f"{rs['skew']:.2f}  {'(right-skewed)' if rs['skew']>0 else '(left-skewed)'}"),
        ("Excess Kurtosis",  f"{rs['kurt']:.2f}  {'(fat tails)' if rs['kurt']>0 else '(thin tails)'}"),
        ("",                 None),
        ("WIN / LOSS",       None),
        ("Win Rate",         f"{rs['win_rate']:.1f}%"),
        ("Avg Win Day",      f"{rs['avg_win']:+.3f}%"),
        ("Avg Loss Day",     f"{rs['avg_loss']:+.3f}%"),
        ("Profit Factor",    f"{rs['pfactor']:.2f}"),
        ("Best Day",         f"{rs['best_day']:+.2f}%"),
        ("Worst Day",        f"{rs['worst_day']:+.2f}%"),
        ("Best Month",       f"{rs['best_month']:+.1f}%"),
        ("Worst Month",      f"{rs['worst_month']:+.1f}%"),
    ]
    for label, value in rows:
        if value is None and label:
            print(f"\n  ── {label} {'─' * (W - len(label) - 6)}")
        elif value is None:
            pass
        else:
            print(f"  {label:<26}{value}")
    print("═" * W)


def plot_strategy_chart(df, rs):
    """Professional strategy performance chart: equity curve, drawdown,
    monthly heatmap, rolling Sharpe, return distribution."""

    DARK_BG  = "#0d1117"
    PANEL_BG = "#161b22"
    GREEN    = "#3fb950"
    RED      = "#f85149"
    BLUE     = "#58a6ff"
    ORANGE   = "#ffa657"
    PURPLE   = "#d2a8ff"
    GREY     = "#8b949e"
    WHITE    = "#e6edf3"
    YELLOW   = "#e3b341"

    fig = plt.figure(figsize=(18, 24))
    fig.patch.set_facecolor(DARK_BG)
    gs = GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.35,
                  height_ratios=[2, 1, 1.6, 1.6])

    def style_ax(ax, title="", xlabel=True):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=GREY, labelsize=8)
        ax.xaxis.label.set_color(GREY)
        ax.yaxis.label.set_color(GREY)
        if title:
            ax.set_title(title, color=WHITE, fontsize=10, pad=8, fontweight="bold")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.grid(axis="y", color="#21262d", lw=0.5, ls="--")
        ax.grid(axis="x", color="#21262d", lw=0.3, ls=":")
        if xlabel:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    dates = df["date"].values
    r     = df["net_yield_daily_pct"].values / 100.0
    ec    = rs["equity_curve"]          # 1→N normalised
    dd    = rs["dd_series"] * 100       # % drawdown (negative)

    # ── 1. Equity curve (log scale, normalised to 100) ───────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ec_idx  = ec * 100                  # base 100
    # benchmark: +0% (flat USDC)
    ax1.semilogy(dates, ec_idx, color=BLUE, lw=2.0, label=f"Strategy (CAGR {rs['cagr']:+.1f}%)")
    ax1.fill_between(dates, ec_idx, 100, where=(ec_idx >= 100), alpha=0.15, color=GREEN)
    ax1.fill_between(dates, ec_idx, 100, where=(ec_idx < 100),  alpha=0.25, color=RED)
    ax1.axhline(100, color=GREY, lw=0.8, ls="--", label="Starting value = 100")
    # Annotate final value
    ax1.annotate(f"{ec_idx[-1]:.0f}", xy=(dates[-1], ec_idx[-1]),
                 xytext=(8, 0), textcoords="offset points",
                 color=BLUE, fontsize=9, fontweight="bold", va="center")
    ax1.set_ylabel("Equity (log scale, base 100)", color=GREY, fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.15, labelcolor=WHITE)
    style_ax(ax1, f"Strategy Equity Curve — Leveraged Basis + stETH  "
                  f"|  Total Return {rs['total_ret']:+.0f}%  ·  CAGR {rs['cagr']:+.1f}%  ·  "
                  f"Sharpe {rs['sharpe']:.2f}  ·  Sortino {rs['sortino']:.2f}  ·  Calmar {rs['calmar']:.2f}")

    # ── 2. Drawdown ───────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(dates, dd, 0, alpha=0.55, color=RED)
    ax2.plot(dates, dd, color=RED, lw=0.8)
    ax2.axhline(rs["max_dd"], color=ORANGE, lw=1.0, ls=":",
                label=f"Max DD  {rs['max_dd']:.1f}%  ({rs['max_dd_dur']} days)")
    ax2.set_ylabel("Drawdown (%)", color=GREY, fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.15, labelcolor=WHITE)
    style_ax(ax2, "Underwater Chart (Drawdown from High-Water Mark)")

    # ── 3. Monthly returns heatmap ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :])
    monthly = rs["monthly"]
    monthly_df = monthly.to_frame("ret")
    monthly_df["year"]  = monthly_df.index.year
    monthly_df["month"] = monthly_df.index.month

    years  = sorted(monthly_df["year"].unique())
    months = list(range(1, 13))
    grid   = np.full((len(years), 12), np.nan)
    for _, row in monthly_df.iterrows():
        yi = years.index(row["year"])
        mi = int(row["month"]) - 1
        grid[yi, mi] = row["ret"]

    # Symmetric color scale capped at ±50% monthly
    vmax = min(50, np.nanpercentile(np.abs(grid), 98))
    im = ax3.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax3.set_xticks(range(12))
    ax3.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"],
                        color=GREY, fontsize=8)
    ax3.set_yticks(range(len(years)))
    ax3.set_yticklabels(years, color=GREY, fontsize=8)
    ax3.tick_params(length=0)
    # Annotate cells
    for yi in range(len(years)):
        for mi in range(12):
            v = grid[yi, mi]
            if not np.isnan(v):
                txt_col = "white" if abs(v) > vmax * 0.5 else "#111"
                ax3.text(mi, yi, f"{v:+.0f}%", ha="center", va="center",
                         fontsize=6.5, color=txt_col, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax3, orientation="vertical", fraction=0.015, pad=0.01)
    cbar.ax.tick_params(colors=GREY, labelsize=7)
    cbar.set_label("Monthly Return (%)", color=GREY, fontsize=8)
    ax3.set_facecolor(PANEL_BG)
    ax3.set_title("Monthly Returns Heatmap (%)", color=WHITE, fontsize=10,
                  pad=8, fontweight="bold")
    for sp in ax3.spines.values():
        sp.set_edgecolor("#30363d")

    # ── 4. Rolling 90-day Sharpe ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3, 0])
    roll_mean = pd.Series(r).rolling(90).mean()
    roll_std  = pd.Series(r).rolling(90).std()
    roll_sh   = (roll_mean / roll_std) * np.sqrt(365)
    roll_sh_c = roll_sh.clip(-5, 10)
    ax4.fill_between(dates, roll_sh_c, 0,
                     where=(roll_sh_c >= 0), alpha=0.20, color=GREEN)
    ax4.fill_between(dates, roll_sh_c, 0,
                     where=(roll_sh_c < 0),  alpha=0.20, color=RED)
    ax4.plot(dates, roll_sh_c, color=PURPLE, lw=1.5,
             label=f"Rolling 90-day Sharpe")
    ax4.axhline(0, color=GREY, lw=0.7, ls="--")
    ax4.axhline(1, color=GREEN, lw=0.7, ls=":", alpha=0.6, label="Sharpe = 1")
    ax4.axhline(rs["sharpe"], color=BLUE, lw=1.0, ls=":",
                label=f"Full-period Sharpe = {rs['sharpe']:.2f}")
    ax4.set_ylabel("Sharpe Ratio", color=GREY, fontsize=9)
    ax4.legend(loc="upper right", fontsize=7.5, framealpha=0.15, labelcolor=WHITE)
    style_ax(ax4, "Rolling 90-Day Sharpe Ratio")

    # ── 5. Return distribution ────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 1])
    r_pct = r * 100
    bins  = np.linspace(np.percentile(r_pct, 0.5), np.percentile(r_pct, 99.5), 80)
    n_vals, bin_edges, patches = ax5.hist(r_pct, bins=bins, color=BLUE,
                                          alpha=0.55, edgecolor="none", density=True)

    # Colour negative bars red
    for patch, left in zip(patches, bin_edges[:-1]):
        if left < 0:
            patch.set_facecolor(RED)
            patch.set_alpha(0.55)

    # Normal overlay
    mu, sigma = r_pct.mean(), r_pct.std()
    x_fit = np.linspace(bins[0], bins[-1], 300)
    from scipy.stats import norm
    ax5.plot(x_fit, norm.pdf(x_fit, mu, sigma), color=ORANGE, lw=1.5,
             ls="--", label=f"Normal(μ={mu:.3f}%, σ={sigma:.3f}%)")

    # VaR lines
    ax5.axvline(rs["var_95"], color=YELLOW, lw=1.2, ls=":",
                label=f"VaR 95%  {rs['var_95']:.3f}%")
    ax5.axvline(rs["var_99"], color=RED, lw=1.2, ls=":",
                label=f"VaR 99%  {rs['var_99']:.3f}%")
    ax5.set_xlabel("Daily Return (%)", color=GREY, fontsize=9)
    ax5.set_ylabel("Density", color=GREY, fontsize=9)
    ax5.legend(loc="upper right", fontsize=7, framealpha=0.15, labelcolor=WHITE)
    style_ax(ax5, f"Daily Return Distribution  |  Skew {rs['skew']:+.2f}  "
                  f"ExKurt {rs['kurt']:+.2f}", xlabel=False)
    ax5.tick_params(axis="x", colors=GREY, labelsize=8)

    # ── Risk table (right half of row 3 — already used, add text block inside ax5's twin) ─
    # Actually place a compact risk block as a text box overlay on ax1
    risk_text = (
        f"  Risk Stats\n"
        f"  ─────────────────\n"
        f"  CAGR         {rs['cagr']:+.1f}%\n"
        f"  Ann. Vol     {rs['ann_vol']:.1f}%\n"
        f"  Sharpe       {rs['sharpe']:.2f}\n"
        f"  Sortino      {rs['sortino']:.2f}\n"
        f"  Calmar       {rs['calmar']:.2f}\n"
        f"  Max DD       {rs['max_dd']:.1f}%\n"
        f"  DD Dur       {rs['max_dd_dur']}d\n"
        f"  VaR 95%      {rs['var_95']:.3f}%\n"
        f"  VaR 99%      {rs['var_99']:.3f}%\n"
        f"  CVaR 95%     {rs['cvar_95']:.3f}%\n"
        f"  CVaR 99%     {rs['cvar_99']:.3f}%\n"
        f"  Win Rate     {rs['win_rate']:.1f}%\n"
        f"  Profit Fac.  {rs['pfactor']:.2f}\n"
        f"  Skew         {rs['skew']:+.2f}\n"
        f"  Ex. Kurt     {rs['kurt']:+.2f}"
    )
    ax1.text(0.998, 0.98, risk_text, transform=ax1.transAxes,
             ha="right", va="top", fontsize=7.5, color=WHITE,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22",
                       edgecolor="#30363d", alpha=0.92))

    fig.suptitle(
        "Leveraged Basis Strategy — Performance & Risk Analysis  |  Jan 2022 – Mar 2025",
        color=WHITE, fontsize=14, fontweight="bold", y=0.998
    )

    plt.savefig("strategy_chart.png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    print("Strategy chart saved → strategy_chart.png")
    plt.close()


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
    rs    = compute_risk_stats(df)

    print_summary(stats)
    print_risk_stats(rs)
    print_quarterly_breakdown(df)
    plot(df, stats)
    plot_strategy_chart(df, rs)

    # Save detailed CSV for inspection
    df.to_csv("backtest_detail.csv", index=False)
    print("\nDetailed daily data saved → backtest_detail.csv")
