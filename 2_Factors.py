import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from data_engine import COLORS

st.set_page_config(page_title="Factor Analysis", page_icon="🧬", layout="wide")

if "portfolio_data" not in st.session_state or st.session_state["portfolio_data"] is None:
    st.warning("No portfolio loaded. Go to the Home page and click **Analyze Portfolio** first.")
    st.stop()

d           = st.session_state["portfolio_data"]
simple_mode = st.session_state.get("simple_mode", True)

DAYS         = 252
factor_betas = d["factor_betas"]
alpha        = d["alpha"]
r2           = d["r_squared"]
tickers      = d["tickers"]
returns      = d["returns"]
ff           = d["ff_factors"]
excess       = d["excess"]

PLAIN_NAMES = {
    "Mkt-RF": "Market Risk",
    "SMB":    "Small vs Large",
    "HML":    "Value vs Growth",
    "RMW":    "Profitability",
    "CMA":    "Investment Style",
}
PLAIN_DESC = {
    "Mkt-RF": "How much you amplify or dampen overall market moves.",
    "SMB":    "Positive = tilts toward small companies. Negative = large companies.",
    "HML":    "Positive = value/cheap stocks. Negative = growth/expensive stocks.",
    "RMW":    "Positive = highly profitable companies. Negative = low-margin firms.",
    "CMA":    "Positive = conservative spenders. Negative = aggressive reinvestors.",
}

st.title("🧬 Factor Analysis")
st.caption(
    "This breaks down *why* your portfolio performs the way it does — "
    "not just what happened, but the underlying bets driving it."
    if simple_mode else
    "Fama-French 5-factor OLS regression on daily excess returns."
)
st.divider()

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    lbl = "How much factors explain" if simple_mode else "R² — Explained Variance"
    st.metric(lbl, f"{r2:.1%}",
              help="What % of your portfolio's moves are explained by these 5 factors.")
with c2:
    lbl = "Extra return beyond factors" if simple_mode else "Alpha (annualized)"
    st.metric(lbl, f"{alpha:+.2%}/yr",
              delta="↑ Positive" if alpha > 0 else "↓ Negative",
              delta_color="normal" if alpha > 0 else "inverse",
              help="Return above what factors alone predict. Positive = outperforming.")
with c3:
    dom = max(factor_betas, key=lambda k: abs(factor_betas[k]))
    lbl = "Biggest driver" if simple_mode else "Dominant Factor"
    st.metric(lbl, PLAIN_NAMES[dom] if simple_mode else dom)

st.divider()

# ── Factor bar chart ──────────────────────────────────────────────────────────
st.subheader("What's Driving Your Returns?" if simple_mode else "Factor Exposures (Beta)")
if simple_mode:
    st.caption("Each bar = how strongly your portfolio leans toward that characteristic. Green = you have it. Red = you lean the other way.")

factors = list(factor_betas.keys())
betas   = list(factor_betas.values())
labels  = [PLAIN_NAMES[f] for f in factors] if simple_mode else factors
colors  = ["#1D9E75" if b >= 0 else "#D85A30" for b in betas]

fig1 = go.Figure(go.Bar(
    x=labels, y=betas,
    marker_color=colors,
    text=[f"{b:+.3f}" for b in betas],
    textposition="outside",
    hovertext=[PLAIN_DESC[f] for f in factors],
    hoverinfo="text+y"
))
fig1.add_hline(y=0, line_color="#888780", line_width=1)
fig1.update_layout(
    yaxis_title="Factor beta",
    title=f"R² = {r2:.0%} of returns explained | Alpha = {alpha:+.2%}/yr above factors",
    height=400, margin=dict(l=60, r=20, t=60, b=80)
)
st.plotly_chart(fig1, use_container_width=True)

# ── Plain English per factor ──────────────────────────────────────────────────
if simple_mode:
    st.subheader("What each factor means for you")
    cols = st.columns(len(factors))
    for i, (factor, beta_val) in enumerate(factor_betas.items()):
        with cols[i]:
            icon = "🟢" if abs(beta_val) > 0.3 else "🟡" if abs(beta_val) > 0.1 else "⚪"
            st.markdown(f"**{PLAIN_NAMES[factor]}**")
            st.markdown(f"{icon} **{beta_val:+.3f}**")
            st.caption(PLAIN_DESC[factor])

st.divider()

# ── Per-stock factor table ─────────────────────────────────────────────────────
st.subheader("Per-Stock Factor Profile" if not simple_mode else "Which Stock Drives What?")
if simple_mode:
    st.caption("How each individual holding contributes to your factor exposures.")

stock_betas = []
for ticker in tickers:
    if ticker not in returns.columns:
        continue
    ret_s = returns[ticker].reindex(excess.index)
    rf_s  = ff["RF"].reindex(excess.index)
    exc_s = (ret_s - rf_s).dropna()
    X_s   = ff[["Mkt-RF","SMB","HML","RMW","CMA"]].reindex(exc_s.index).dropna()
    exc_s = exc_s.reindex(X_s.index)
    if len(exc_s) < 60:
        continue
    try:
        m_s  = LinearRegression().fit(X_s.values, exc_s.values)
        row  = {"Ticker": ticker}
        row.update(dict(zip(["Mkt-RF","SMB","HML","RMW","CMA"], m_s.coef_)))
        row["Alpha"] = f"{m_s.intercept_ * DAYS:+.2%}/yr"
        row["R²"]    = f"{m_s.score(X_s.values, exc_s.values):.0%}"
        stock_betas.append(row)
    except Exception:
        pass

if stock_betas:
    df_stocks = pd.DataFrame(stock_betas).set_index("Ticker")
    if simple_mode:
        df_display = df_stocks.copy()
        num_cols = [c for c in df_display.columns if c not in ("Alpha","R²")]
        df_display.columns = [PLAIN_NAMES.get(c, c) for c in df_display.columns]
        st.dataframe(df_display.round(3), use_container_width=True)
    else:
        st.dataframe(df_stocks.round(3), use_container_width=True)

    if simple_mode and len(stock_betas) >= 2:
        mkt_col = "Mkt-RF"
        highest = df_stocks[mkt_col].idxmax()
        lowest  = df_stocks[mkt_col].idxmin()
        st.info(
            f"**{highest}** amplifies market moves the most (highest market beta).  \n"
            f"**{lowest}** is the most defensive holding in your portfolio."
        )
