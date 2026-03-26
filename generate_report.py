"""
generate_report.py
==================
Produces a professional multi-page PDF report for the Leveraged Basis Strategy backtest.

Pages:
  1. Cover
  2. Strategy Description
  3. Risk Statistics & Key Metrics
  4. Cumulative P&L chart  (backtest_results.png)
  5. Strategy Performance chart  (strategy_chart.png)
  6. $10k Portfolio Simulation chart  (portfolio_chart.png)
  7. Quarterly breakdown table
  8. Monthly portfolio P&L table

Run:
    python generate_report.py
"""

import sys
import textwrap
from datetime import date

import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)
from reportlab.platypus.flowables import BalancedColumns
from reportlab.lib.colors import HexColor, white, black

# ── Colour palette (GitHub dark theme inspired) ──────────────────────────────
C_BG       = HexColor("#0d1117")
C_PANEL    = HexColor("#161b22")
C_GREEN    = HexColor("#3fb950")
C_RED      = HexColor("#f85149")
C_BLUE     = HexColor("#58a6ff")
C_ORANGE   = HexColor("#ffa657")
C_PURPLE   = HexColor("#d2a8ff")
C_GREY     = HexColor("#8b949e")
C_WHITE    = HexColor("#e6edf3")
C_YELLOW   = HexColor("#e3b341")
C_DARKGREY = HexColor("#30363d")
C_HEADING  = HexColor("#1f6feb")

PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm

# ── Styles ───────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

COVER_TITLE = S("CoverTitle", fontSize=32, leading=40, textColor=C_WHITE,
                alignment=TA_CENTER, fontName="Helvetica-Bold")
COVER_SUB   = S("CoverSub", fontSize=14, leading=20, textColor=C_GREY,
                alignment=TA_CENTER, fontName="Helvetica")
COVER_META  = S("CoverMeta", fontSize=11, leading=16, textColor=C_BLUE,
                alignment=TA_CENTER, fontName="Helvetica")

H1 = S("H1", fontSize=18, leading=24, textColor=C_WHITE,
        fontName="Helvetica-Bold", spaceAfter=6)
H2 = S("H2", fontSize=13, leading=18, textColor=C_BLUE,
        fontName="Helvetica-Bold", spaceAfter=4, spaceBefore=10)
BODY = S("Body", fontSize=9, leading=14, textColor=C_WHITE,
         fontName="Helvetica", spaceAfter=4)
BODY_GREY = S("BodyGrey", fontSize=8.5, leading=13, textColor=C_GREY,
              fontName="Helvetica", spaceAfter=3)
LABEL = S("Label", fontSize=8, leading=11, textColor=C_GREY,
          fontName="Helvetica-Bold")
VALUE_GREEN = S("ValGreen", fontSize=9, leading=12, textColor=C_GREEN,
                fontName="Helvetica-Bold")
VALUE_RED   = S("ValRed",   fontSize=9, leading=12, textColor=C_RED,
                fontName="Helvetica-Bold")
VALUE_WHITE = S("ValWhite", fontSize=9, leading=12, textColor=C_WHITE,
                fontName="Helvetica-Bold")
FOOTER_ST = S("Footer", fontSize=7, leading=9, textColor=C_GREY,
              fontName="Helvetica", alignment=TA_CENTER)
CAPTION = S("Caption", fontSize=8, leading=11, textColor=C_GREY,
            fontName="Helvetica", alignment=TA_CENTER, spaceAfter=6)

# ── Page template with dark background + header/footer ───────────────────────
class DarkCanvas:
    def __init__(self, title="Leveraged Basis Strategy — Backtest Report"):
        self.title = title
        self._page = 0

    def __call__(self, canvas, doc):
        self._page += 1
        w, h = A4
        canvas.saveState()

        # Full-page dark background
        canvas.setFillColor(C_BG)
        canvas.rect(0, 0, w, h, fill=1, stroke=0)

        if self._page > 1:
            # Top rule
            canvas.setStrokeColor(C_DARKGREY)
            canvas.setLineWidth(0.5)
            canvas.line(MARGIN, h - 1.1*cm, w - MARGIN, h - 1.1*cm)
            # Header text
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(C_GREY)
            canvas.drawString(MARGIN, h - 0.9*cm, self.title)
            canvas.drawRightString(w - MARGIN, h - 0.9*cm,
                                   f"Confidential — {date.today().strftime('%d %b %Y')}")
            # Footer rule + page number
            canvas.line(MARGIN, 0.9*cm, w - MARGIN, 0.9*cm)
            canvas.drawCentredString(w / 2, 0.55*cm, f"Page {self._page - 1}")

        canvas.restoreState()


# ── Helper: coloured stat cell ────────────────────────────────────────────────
def stat_cell(label, value, colour=None):
    if colour is None:
        if value.startswith("+"):
            colour = C_GREEN
        elif value.startswith("−") or value.startswith("-"):
            colour = C_RED
        else:
            colour = C_WHITE
    vstyle = ParagraphStyle("_vc", fontSize=11, leading=14,
                             textColor=colour, fontName="Helvetica-Bold",
                             alignment=TA_CENTER)
    lstyle = ParagraphStyle("_lc", fontSize=7.5, leading=10,
                             textColor=C_GREY, fontName="Helvetica",
                             alignment=TA_CENTER)
    return [Paragraph(value, vstyle), Paragraph(label, lstyle)]


def kpi_table(pairs, col_count=4):
    """pairs = list of (label, value_str). Renders as a dark KPI grid."""
    cells = [stat_cell(l, v) for l, v in pairs]
    # pad to full rows
    while len(cells) % col_count:
        cells.append(["", ""])
    rows = [cells[i:i+col_count] for i in range(0, len(cells), col_count)]
    col_w = (PAGE_W - 2*MARGIN) / col_count
    t = Table(rows, colWidths=[col_w]*col_count, rowHeights=None)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), C_PANEL),
        ("BOX",         (0,0), (-1,-1), 0.5, C_DARKGREY),
        ("INNERGRID",   (0,0), (-1,-1), 0.3, C_DARKGREY),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
    ]))
    return t


def section_rule(title):
    return [
        Spacer(1, 0.35*cm),
        HRFlowable(width="100%", thickness=0.5, color=C_HEADING, spaceAfter=4),
        Paragraph(title, H2),
    ]


# ── Data helpers (run backtest inline) ───────────────────────────────────────
def load_data():
    """Re-run the backtest to get all numbers without re-importing the module."""
    sys.path.insert(0, ".")
    import backtest as bt
    df        = bt.run_backtest()
    stats     = bt.compute_stats(df)
    rs        = bt.compute_risk_stats(df)
    daily_p, monthly_df, quarterly_df = bt.compute_portfolio_pnl(df, 10_000)
    return df, stats, rs, daily_p, monthly_df, quarterly_df, bt


# ── Build PDF ─────────────────────────────────────────────────────────────────
def build_pdf(output="report.pdf"):
    print("Loading backtest data …")
    df, stats, rs, daily_p, monthly_df, quarterly_df, bt = load_data()
    lev = bt.leverage_multiplier

    story = []
    on_page = DarkCanvas()

    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=1.6*cm, bottomMargin=1.4*cm,
        title="Leveraged Basis Strategy — Backtest Report",
        author="LeveragedBasis",
    )

    img_w = PAGE_W - 2*MARGIN   # full-width images

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — COVER
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 3.5*cm))
    story.append(Paragraph("LEVERAGED BASIS STRATEGY", COVER_TITLE))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Backtest Report  ·  Jan 2022 – Mar 2026", COVER_SUB))
    story.append(Spacer(1, 0.6*cm))
    story.append(HRFlowable(width="60%", thickness=1, color=C_HEADING,
                             hAlign="CENTER", spaceAfter=8))
    story.append(Spacer(1, 0.4*cm))

    # Hero KPIs on cover
    cover_kpis = [
        ("Total Return",      f"+{rs['total_ret']:.0f}%"),
        ("CAGR",              f"+{rs['cagr']:.1f}%"),
        ("Sharpe Ratio",      f"{rs['sharpe']:.2f}"),
        ("Max Drawdown",      f"{rs['max_dd']:.1f}%"),
        ("$10k → Final",      f"${daily_p['portfolio_value'].iloc[-1]:,.0f}"),
        ("Net APR (avg)",     f"+{stats['avg_net_apr']:.1f}%"),
        ("Sortino Ratio",     f"{rs['sortino']:.2f}"),
        ("Calmar Ratio",      f"{rs['calmar']:.2f}"),
    ]
    story.append(kpi_table(cover_kpis, col_count=4))
    story.append(Spacer(1, 1.2*cm))
    story.append(Paragraph(
        f"Strategy: {bt.N_LOOPS}× ETH Loop  ·  {bt.LTV_PER_LOOP:.0%} LTV/loop  ·  "
        f"×{lev:.2f} leverage  ·  Delta-Neutral via ETH-USDT Perp Short  ·  stETH staking 2.7%",
        COVER_META))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        f"Prepared: {date.today().strftime('%d %B %Y')}  ·  Confidential",
        COVER_SUB))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — STRATEGY DESCRIPTION
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Strategy Overview", H1))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_HEADING, spaceAfter=8))

    desc_paras = [
        ("What is it?",
         "The Leveraged Basis Strategy is a <b>delta-neutral carry trade</b> on Ethereum. "
         "It earns yield from two independent sources — perpetual funding rates and stETH "
         "staking rewards — while remaining fully hedged against ETH price movements."),

        ("How it works",
         "1. <b>Collateral loop (5×):</b> Starting with 1 stETH, the strategy repeatedly "
         "deposits stETH, borrows USDC at 80% LTV, buys more stETH, and re-deposits. "
         f"After 5 loops this creates a ×{lev:.2f} leveraged stETH position on-chain.<br/><br/>"
         "2. <b>Delta hedge:</b> An equal-sized short position on the ETH-USDT perpetual "
         "futures (Binance) cancels all directional ETH exposure. The portfolio is "
         "<b>price-neutral</b> — it neither gains nor loses from ETH price moves.<br/><br/>"
         "3. <b>Revenue streams:</b><br/>"
         f"&nbsp;&nbsp;• <b>Funding income:</b> Perpetual longs pay shorts when the market "
         f"is bullish (avg raw rate {stats['avg_raw_funding']:.1f}% APR). "
         f"At ×{lev:.2f} leverage this amplifies to ~{stats['avg_funding_apr']:.0f}% APR on equity.<br/>"
         f"&nbsp;&nbsp;• <b>Staking income:</b> All looped stETH earns ~2.7% APR, "
         f"amplified to ~{stats['avg_staking_apr']:.1f}% APR on equity.<br/><br/>"
         "4. <b>Cost:</b> USDC borrow interest on the leveraged position "
         f"(avg {stats['avg_raw_borrow']:.1f}% APR including 0.5% protocol spread). "
         f"This nets to ~{stats['avg_borrow_apr']:.0f}% APR drag on equity."),

        ("Quarterly rebalancing",
         "At the end of each quarter the position is rebalanced: collateral, short notional, "
         "and borrow size are all rescaled to the current portfolio value. This ensures "
         "the leverage ratio stays constant and compounding gains are fully deployed."),

        ("Key risks",
         "• <b>Negative funding:</b> When the market turns bearish, longs pay nothing and "
         "shorts may pay longs (2022Q3: −53.6% annualised; 2025Q1: −35.5%).<br/>"
         "• <b>Borrow rate spikes:</b> DeFi utilisation surges can push USDC rates above "
         "funding income (notable in late 2023 when borrow cost hit ~20% annualised).<br/>"
         "• <b>Liquidation risk:</b> 80% LTV per loop leaves a buffer, but extreme "
         "price volatility or oracle manipulation could threaten collateral ratios.<br/>"
         "• <b>Smart-contract risk:</b> Aave/Compound protocol failures or exploits.<br/>"
         "• <b>Execution slippage:</b> Modelled at 10% funding take-rate."),

        ("Backtest assumptions",
         f"Period: Jan 2022 – Mar 2026 ({stats['years']:.1f} years) · "
         f"Loops: {bt.N_LOOPS} · LTV: {bt.LTV_PER_LOOP:.0%}/loop · "
         f"Leverage: ×{lev:.2f} · Funding take-rate: {bt.FUNDING_TAKE_RATE:.0%} · "
         f"Borrow spread: +{bt.BORROW_SPREAD*100:.1f}% · Staking APR: {bt.STAKING_APR*100:.1f}% · "
         "Data: Binance ETH-USDT 8h funding; Aave/Compound USDC borrow; CoinGecko ETH price."),
    ]

    for heading, body in desc_paras:
        story += section_rule(heading)
        story.append(Paragraph(body, BODY))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — RISK STATISTICS
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Risk Statistics", H1))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_HEADING, spaceAfter=8))

    risk_kpis = [
        ("Total Return",      f"+{rs['total_ret']:.1f}%"),
        ("CAGR",              f"+{rs['cagr']:.1f}%"),
        ("Ann. Volatility",   f"{rs['ann_vol']:.1f}%"),
        ("Sharpe Ratio",      f"{rs['sharpe']:.2f}"),
        ("Sortino Ratio",     f"{rs['sortino']:.2f}"),
        ("Calmar Ratio",      f"{rs['calmar']:.2f}"),
        ("Max Drawdown",      f"{rs['max_dd']:.1f}%"),
        ("Max DD Duration",   f"{rs['max_dd_dur']} days"),
        ("VaR 95% (daily)",   f"{rs['var_95']:.3f}%"),
        ("VaR 99% (daily)",   f"{rs['var_99']:.3f}%"),
        ("CVaR 95%",          f"{rs['cvar_95']:.3f}%"),
        ("CVaR 99%",          f"{rs['cvar_99']:.3f}%"),
        ("Win Rate",          f"{rs['win_rate']:.1f}%"),
        ("Profit Factor",     f"{rs['pfactor']:.2f}"),
        ("Skewness",          f"{rs['skew']:+.2f}"),
        ("Excess Kurtosis",   f"{rs['kurt']:+.2f}"),
        ("Best Day",          f"+{rs['best_day']:.2f}%"),
        ("Worst Day",         f"{rs['worst_day']:+.2f}%"),
        ("Best Month",        f"+{rs['best_month']:.1f}%"),
        ("Worst Month",       f"{rs['worst_month']:+.1f}%"),
    ]
    story.append(kpi_table(risk_kpis, col_count=4))
    story.append(Spacer(1, 0.5*cm))

    # P&L summary row
    pnl_kpis = [
        ("Total Funding Income", f"+${stats['total_funding_usd']:,.0f}"),
        ("Total Staking Income", f"+${stats['total_staking_usd']:,.0f}"),
        ("Total Borrow Cost",    f"−${stats['total_borrow_usd']:,.0f}"),
        ("Total Net P&L",        f"+${stats['total_net_usd']:,.0f}"),
        ("Avg Funding APR",      f"+{stats['avg_funding_apr']:.1f}%"),
        ("Avg Staking APR",      f"+{stats['avg_staking_apr']:.1f}%"),
        ("Avg Borrow APR",       f"−{stats['avg_borrow_apr']:.1f}%"),
        ("Avg Net APR",          f"+{stats['avg_net_apr']:.1f}%"),
    ]
    story += section_rule("P&L Attribution (per 1 stETH collateral)")
    story.append(kpi_table(pnl_kpis, col_count=4))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 4 — CUMULATIVE P&L CHART
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Cumulative P&L Breakdown", H1))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_HEADING, spaceAfter=6))
    story.append(Paragraph(
        "Funding income (green), staking income (purple), borrow cost (red) and net P&L "
        "(blue) — all cumulative USD per 1 stETH initial collateral.", BODY_GREY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Image("backtest_results.png", width=img_w, height=img_w * 22/18))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 5 — STRATEGY PERFORMANCE CHART
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Strategy Performance & Risk Analysis", H1))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_HEADING, spaceAfter=6))
    story.append(Paragraph(
        "Equity curve (log scale), underwater drawdown chart, monthly return heatmap, "
        "rolling 90-day Sharpe ratio, and daily return distribution.", BODY_GREY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Image("strategy_chart.png", width=img_w, height=img_w * 24/18))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 6 — $10k PORTFOLIO CHART
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("$10,000 USDC Portfolio Simulation", H1))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_HEADING, spaceAfter=6))
    final_val = daily_p["portfolio_value"].iloc[-1]
    story.append(Paragraph(
        f"Starting with $10,000 USDC, daily compounding of net yield, "
        f"position rebalanced at each quarter-end.  "
        f"Final value: <b>${final_val:,.0f}</b>  ·  "
        f"Total return: <b>+{(final_val/10000-1)*100:.0f}%</b>  ·  "
        f"CAGR: <b>+{((final_val/10000)**(365.25/len(daily_p))-1)*100:.1f}%</b>", BODY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Image("portfolio_chart.png", width=img_w, height=img_w * 20/18))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 7 — QUARTERLY BREAKDOWN TABLE
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Quarterly Breakdown", H1))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_HEADING, spaceAfter=6))
    story.append(Paragraph(
        "Annualised % return on equity per quarter (per 1 stETH collateral). "
        "Borrow cost shown as negative.", BODY_GREY))
    story.append(Spacer(1, 0.2*cm))

    q_headers = ["Quarter", "Funding APR", "Staking APR", "Borrow Cost",
                 "Net APR", "Gross Income", "Q Return ($10k)", "Days"]
    q_rows = [q_headers]

    df2 = df.copy()
    df2["year"]    = df2["date"].dt.year
    df2["quarter"] = df2["date"].dt.quarter

    # also get portfolio quarterly returns
    qp_lookup = {}
    for _, row in quarterly_df.iterrows():
        qp_lookup[(int(row["year"]), int(row["quarter"]))] = row["q_ret"]

    for (yr, q), g in df2.groupby(["year", "quarter"]):
        n = len(g)
        ann = 365.0 / n
        fa  =  g["funding_yield_daily_pct"].sum() * ann
        sa  =  g["staking_yield_daily_pct"].sum() * ann
        bc  = -g["borrow_yield_daily_pct"].sum()  * ann
        na  =  g["net_yield_daily_pct"].sum()      * ann
        gr  = fa + sa
        qr  = qp_lookup.get((yr, q), 0.0)
        q_rows.append([
            f"{yr}Q{q}",
            f"{fa:+.1f}%",
            f"{sa:+.1f}%",
            f"{bc:+.1f}%",
            f"{na:+.1f}%",
            f"{gr:+.1f}%",
            f"{qr:+.1f}%",
            str(n),
        ])

    # totals
    ann_all = 365.0 / len(df2)
    q_rows.append([
        "TOTAL",
        f"{df2['funding_yield_daily_pct'].sum()*ann_all:+.1f}%",
        f"{df2['staking_yield_daily_pct'].sum()*ann_all:+.1f}%",
        f"{-df2['borrow_yield_daily_pct'].sum()*ann_all:+.1f}%",
        f"{df2['net_yield_daily_pct'].sum()*ann_all:+.1f}%",
        f"{(df2['funding_yield_daily_pct'].sum()+df2['staking_yield_daily_pct'].sum())*ann_all:+.1f}%",
        f"+{(final_val/10000-1)*100:.1f}%",
        str(len(df2)),
    ])

    col_w_q = (PAGE_W - 2*MARGIN) / len(q_headers)
    qt = Table(q_rows, colWidths=[col_w_q]*len(q_headers), repeatRows=1)

    def qrow_style(row_idx, na_val):
        if na_val > 0:
            return colors.HexColor("#1a3a1a")
        elif na_val < 0:
            return colors.HexColor("#3a1a1a")
        return C_PANEL

    ts_q = [
        ("BACKGROUND",  (0,0), (-1,0), C_HEADING),
        ("TEXTCOLOR",   (0,0), (-1,0), white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 7.5),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ("BOX",         (0,0), (-1,-1), 0.5, C_DARKGREY),
        ("INNERGRID",   (0,0), (-1,-1), 0.3, C_DARKGREY),
        ("BACKGROUND",  (0, len(q_rows)-1), (-1, len(q_rows)-1), C_HEADING),
        ("TEXTCOLOR",   (0, len(q_rows)-1), (-1, len(q_rows)-1), white),
        ("FONTNAME",    (0, len(q_rows)-1), (-1, len(q_rows)-1), "Helvetica-Bold"),
    ]
    # colour data rows by net APR sign (col index 4 = Net APR)
    for i, row in enumerate(q_rows[1:-1], start=1):
        try:
            val = float(row[4].replace("%","").replace("+",""))
        except:
            val = 0
        bg = colors.HexColor("#1a3a1a") if val > 0 else colors.HexColor("#3a1a1a")
        ts_q.append(("BACKGROUND", (0,i), (-1,i), bg))
        tc = C_GREEN if val > 0 else C_RED
        ts_q.append(("TEXTCOLOR", (0,i), (-1,i), tc))

    qt.setStyle(TableStyle(ts_q))
    story.append(qt)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 8 — MONTHLY PORTFOLIO P&L TABLE
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Monthly Portfolio P&L — $10,000 USDC", H1))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_HEADING, spaceAfter=6))
    story.append(Paragraph(
        "Daily compounding, position rebalanced at each quarter-end. "
        "Negative months highlighted red, positive green.", BODY_GREY))
    story.append(Spacer(1, 0.2*cm))

    MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    m_headers = ["Month", "Start ($)", "End ($)", "P&L ($)", "Return %",
                 "Net APR %", "Gross APR %", "Days"]
    m_rows = [m_headers]
    prev_q = None
    for _, row in monthly_df.iterrows():
        mo_name = f"{MONTH_NAMES[int(row['month'])-1]} {int(row['year'])}"
        gross   = row["fund_apr"] + row["stake_apr"]
        m_rows.append([
            mo_name,
            f"${row['start_val']:,.0f}",
            f"${row['end_val']:,.0f}",
            f"{'+'if row['mo_pnl']>=0 else ''}${row['mo_pnl']:,.0f}",
            f"{row['mo_ret']:+.2f}%",
            f"{row['net_apr']:+.1f}%",
            f"{gross:+.1f}%",
            str(int(row["n_days"])),
        ])

    col_w_m = (PAGE_W - 2*MARGIN) / len(m_headers)
    mt = Table(m_rows, colWidths=[col_w_m]*len(m_headers), repeatRows=1)

    ts_m = [
        ("BACKGROUND",   (0,0), (-1,0), C_HEADING),
        ("TEXTCOLOR",    (0,0), (-1,0), white),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 7),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0), (-1,-1), 2.5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 2.5),
        ("BOX",          (0,0), (-1,-1), 0.5, C_DARKGREY),
        ("INNERGRID",    (0,0), (-1,-1), 0.3, C_DARKGREY),
    ]
    for i, row in enumerate(m_rows[1:], start=1):
        try:
            ret = float(row[4].replace("%","").replace("+",""))
        except:
            ret = 0
        bg = colors.HexColor("#1a3a1a") if ret > 0 else colors.HexColor("#3a1a1a")
        tc = C_GREEN if ret > 0 else C_RED
        ts_m.append(("BACKGROUND", (0,i), (-1,i), bg))
        ts_m.append(("TEXTCOLOR",  (0,i), (-1,i), tc))
    mt.setStyle(TableStyle(ts_m))
    story.append(mt)

    # ── Build ────────────────────────────────────────────────────────────────
    print("Building PDF …")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Report saved → {output}")


if __name__ == "__main__":
    build_pdf("report.pdf")
