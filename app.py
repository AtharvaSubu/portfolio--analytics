import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from data_engine import load_portfolio_data, validate_tickers, COLORS

st.set_page_config(
    page_title  = "Portfolio Analytics",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Sidebar — portfolio input ─────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Portfolio Analytics")
    st.caption("Institutional-grade analysis for every investor")
    st.divider()

    mode = st.radio(
        "Display mode",
        ["Simple — plain English", "Advanced — raw numbers"],
        index=0,
    )
    st.session_state["simple_mode"] = mode.startswith("Simple")

    st.divider()
    st.subheader("Your Portfolio")
    st.caption("Enter tickers and weights below. Weights must add up to 100%.")

    # Default portfolio
    default_tickers = "AAPL, MSFT, TSLA, JPM, XOM"
    default_weights = "25, 20, 15, 20, 20"

    ticker_input = st.text_input(
        "Tickers (comma separated)",
        value=default_tickers,
        placeholder="e.g. AAPL, GOOGL, MSFT",
        help="Use Yahoo Finance ticker symbols. US stocks: AAPL. Indian stocks: RELIANCE.NS"
    )
    weight_input = st.text_input(
        "Weights % (comma separated)",
        value=default_weights,
        placeholder="e.g. 30, 30, 40",
        help="Must add up to 100"
    )

    analyze_btn = st.button("🔍 Analyze Portfolio", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        "💡 **Tip:** Use Yahoo Finance symbols\n\n"
        "🇺🇸 US stocks: `AAPL`, `NVDA`\n\n"
        "🇮🇳 Indian stocks: `RELIANCE.NS`\n\n"
        "🇬🇧 UK stocks: `BP.L`\n\n"
        "📊 ETFs: `SPY`, `QQQ`"
    )

# ── Parse inputs ──────────────────────────────────────────────────────────────
def parse_inputs(ticker_str, weight_str):
    tickers = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    try:
        weights_raw = [float(w.strip()) for w in weight_str.split(",") if w.strip()]
    except ValueError:
        return None, None, "Weights must be numbers separated by commas."

    if len(tickers) != len(weights_raw):
        return None, None, f"You entered {len(tickers)} tickers but {len(weights_raw)} weights. They must match."

    if len(tickers) < 2:
        return None, None, "Please enter at least 2 tickers."

    total = sum(weights_raw)
    if abs(total - 100) > 0.5:
        return None, None, f"Weights add up to {total:.1f}% — they must add up to 100%."

    weights_norm = [w / total for w in weights_raw]
    portfolio    = dict(zip(tickers, weights_norm))
    return tickers, portfolio, None


# ── Initialize session state ──────────────────────────────────────────────────
if "portfolio_data" not in st.session_state:
    st.session_state["portfolio_data"] = None
if "error"          not in st.session_state:
    st.session_state["error"] = None

# ── On button click: fetch data ───────────────────────────────────────────────
if analyze_btn:
    tickers, portfolio, parse_error = parse_inputs(ticker_input, weight_input)

    if parse_error:
        st.session_state["error"] = parse_error
        st.session_state["portfolio_data"] = None
    else:
        st.session_state["error"] = None
        with st.spinner("Fetching live data from Yahoo Finance... this takes ~15 seconds"):
            try:
                data = load_portfolio_data(tuple(portfolio.items()))
                st.session_state["portfolio_data"] = data
            except Exception as e:
                st.session_state["error"] = f"Could not load data: {e}"
                st.session_state["portfolio_data"] = None

# ── Show errors ───────────────────────────────────────────────────────────────
if st.session_state["error"]:
    st.error(st.session_state["error"])

# ── Landing page (no data yet) ────────────────────────────────────────────────
if st.session_state["portfolio_data"] is None and not st.session_state["error"]:
    st.title("📊 Portfolio Analytics Platform")
    st.markdown(
        "**Enter your portfolio in the sidebar and click Analyze Portfolio.**\n\n"
        "You'll get institutional-grade analysis in seconds — no code, no spreadsheets."
    )
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📉 Risk Dashboard")
        st.markdown(
            "Return, volatility, Sharpe ratio, drawdown, VaR, "
            "correlation heatmap, sector allocation."
        )
    with col2:
        st.markdown("### 🧬 Factor Analysis")
        st.markdown(
            "Fama-French 5-factor decomposition — understand *why* your "
            "portfolio performs the way it does."
        )
    with col3:
        st.markdown("### 💱 FX Analysis")
        st.markdown(
            "Hidden currency risk in your equity holdings, "
            "plus live FX pair analysis and carry trade scoring."
        )
    st.divider()
    st.info("👈 Enter your tickers and weights in the sidebar to get started.")
    st.stop()

# ── Main dashboard (data loaded) ──────────────────────────────────────────────
d           = st.session_state["portfolio_data"]
simple_mode = st.session_state.get("simple_mode", True)

ann_ret      = d["ann_return"]
ann_vol      = d["ann_vol"]
sharpe       = d["sharpe"]
max_dd       = d["max_dd"]
var_95       = d["var_95"]
beta         = d["beta"]
sp500_ann    = d["sp500_ann"]
sp500_vol    = d["sp500_vol"]
sp500_sharpe = d["sp500_sharpe"]

tickers  = d["tickers"]
weights  = d["weights"]
portfolio= d["portfolio"]

# Scoring helpers
def sharpe_label(s):
    if s > 2:   return "Exceptional", 10
    if s > 1.5: return "Excellent", 9
    if s > 1:   return "Good", 7
    if s > 0.5: return "Fair", 5
    if s > 0:   return "Poor", 3
    return "Very poor", 1

def var_label(v):
    p = abs(v)
    if p < 0.03: return "Very low risk", 10
    if p < 0.05: return "Low risk", 8
    if p < 0.08: return "Moderate risk", 6
    if p < 0.12: return "High risk", 4
    return "Very high risk", 2

def dd_label(dd):
    p = abs(dd)
    if p < 0.10: return "Minimal", 10
    if p < 0.20: return "Manageable", 8
    if p < 0.35: return "Significant", 5
    if p < 0.50: return "Severe", 3
    return "Extreme", 1

sh_lbl, sh_score = sharpe_label(sharpe)
vr_lbl, vr_score = var_label(var_95)
dd_lbl, dd_score = dd_label(max_dd)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Portfolio Overview")
st.caption(
    f"Analysis for: **{', '.join(tickers)}** | "
    f"Data: {d['start_date']} to {d['end_date']} | "
    f"Source: Yahoo Finance (live)"
)
st.divider()

# ── Holdings table ────────────────────────────────────────────────────────────
with st.expander("📋 Your Holdings", expanded=False):
    rows = []
    for t, w in zip(tickers, weights):
        info = d["fundamentals"].get(t, {})
        rows.append({
            "Ticker":   t,
            "Company":  info.get("company", t),
            "Sector":   info.get("sector", "Unknown"),
            "Weight":   f"{w:.1%}",
            "Mkt Cap":  f"${info.get('market_cap',0)/1e9:.0f}B" if info.get("market_cap",0) > 0 else "—",
            "P/E":      f"{info.get('pe_ratio',0):.1f}" if info.get("pe_ratio",0) > 0 else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── KPI metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    delta = f"{ann_ret - sp500_ann:+.1%} vs S&P"
    dc    = "normal" if ann_ret > sp500_ann else "inverse"
    st.metric("Annual Return", f"{ann_ret:.1%}", delta=delta, delta_color=dc)

with c2:
    delta = f"{ann_vol - sp500_vol:+.1%} vs S&P"
    dc    = "inverse" if ann_vol > sp500_vol else "normal"
    st.metric("Volatility" if not simple_mode else "Risk Level", f"{ann_vol:.1%}",
              delta=delta, delta_color=dc)

with c3:
    label = f"Quality ({sh_score}/10)" if simple_mode else "Sharpe Ratio"
    delta = f"{sharpe - sp500_sharpe:+.2f} vs S&P"
    dc    = "normal" if sharpe > sp500_sharpe else "inverse"
    st.metric(label, f"{sharpe:.2f}", delta=delta, delta_color=dc)

with c4:
    label = f"Worst Drop ({dd_score}/10)" if simple_mode else "Max Drawdown"
    st.metric(label, f"{max_dd:.1%}")

with c5:
    label = f"Worst Month ({vr_score}/10)" if simple_mode else "VaR 95%"
    st.metric(label, f"{var_95:.1%}")

st.divider()

# ── Plain English summary ─────────────────────────────────────────────────────
if simple_mode:
    beats = ann_ret > sp500_ann
    st.info(
        f"Your portfolio **{'outperforms' if beats else 'underperforms'}** the S&P 500 "
        f"by **{abs(ann_ret - sp500_ann):.1%}** per year.  \n\n"
        f"Return quality is **{sh_lbl}** (Sharpe: {sharpe:.2f} — "
        f"{'above' if sharpe > sp500_sharpe else 'below'} the S&P 500's {sp500_sharpe:.2f}).  \n\n"
        f"In a bad month (1-in-20), you could lose up to **{abs(var_95):.1%}**.  \n\n"
        f"Your worst historical drop was **{max_dd:.1%}** — classified as **{dd_lbl}**."
    )

# ── Cumulative returns chart ───────────────────────────────────────────────────
st.subheader("Growth of $10,000" if simple_mode else "Cumulative Returns")

cum_port  = d["cum_port"]
cum_sp500 = d["cum_sp500"]
total_p   = (cum_port.iloc[-1]  - 1) * 100
total_s   = (cum_sp500.iloc[-1] - 1) * 100

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=cum_port.index, y=cum_port.values * 10000,
    name=f"Your Portfolio ({total_p:+.0f}%)",
    line=dict(color="#7F77DD", width=2.5)
))
fig.add_trace(go.Scatter(
    x=cum_sp500.index, y=cum_sp500.values * 10000,
    name=f"S&P 500 ({total_s:+.0f}%)",
    line=dict(color="#888780", width=1.5, dash="dot")
))
fig.update_layout(
    yaxis_title="Portfolio value ($)",
    height=380, hovermode="x unified",
    margin=dict(l=60, r=20, t=30, b=40),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)")
)
st.plotly_chart(fig, use_container_width=True)

if simple_mode:
    final_val = cum_port.iloc[-1] * 10000
    bench_val = cum_sp500.iloc[-1] * 10000
    st.caption(
        f"$10,000 invested 5 years ago → "
        f"**Your portfolio: ${final_val:,.0f}** vs S&P 500: ${bench_val:,.0f}"
    )

st.divider()
st.caption("👈 Use the **sidebar pages** to explore Risk, Factors, and FX analysis in depth.")
