import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from data_engine import COLORS

st.set_page_config(page_title="Risk Dashboard", page_icon="📉", layout="wide")

# ── Guard: redirect if no data ────────────────────────────────────────────────
if "portfolio_data" not in st.session_state or st.session_state["portfolio_data"] is None:
    st.warning("No portfolio loaded. Please go to the Home page and click **Analyze Portfolio** first.")
    st.stop()

d           = st.session_state["portfolio_data"]
simple_mode = st.session_state.get("simple_mode", True)

DAYS     = 252
port     = d["port_ret"]
sp500    = d["sp500_ret"]
ff       = d["ff_factors"]
tickers  = d["tickers"]
weights  = d["weights"]
returns  = d["returns"]

ann_ret      = d["ann_return"]
ann_vol      = d["ann_vol"]
sharpe       = d["sharpe"]
max_dd       = d["max_dd"]
var_95       = d["var_95"]
beta         = d["beta"]
sp500_ann    = d["sp500_ann"]
sp500_vol    = d["sp500_vol"]
sp500_sharpe = d["sp500_sharpe"]
ann_rf       = d["ann_rf"]

cum_port       = d["cum_port"]
cum_sp500      = d["cum_sp500"]
drawdown       = d["drawdown"]
monthly        = d["monthly_ret"]
rolling_sharpe = d["rolling_sharpe"]
corr           = d["corr_matrix"]
sector_wt      = d["sector_weights"]

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
st.title("📉 Risk Dashboard")
st.caption(
    "How your portfolio performed and how much risk you took to get there. "
    "Toggle Simple / Advanced in the sidebar." if simple_mode else
    "Full risk analytics with benchmark comparison."
)
st.divider()

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    lbl = f"Quality Score: {sh_score}/10 — {sh_lbl}" if simple_mode else "Sharpe Ratio"
    st.metric(lbl, f"{sharpe:.2f}",
              help="Return earned per unit of risk. Above 1.0 is good, above 2.0 is excellent.")
with c2:
    lbl = f"Risk Level: {vr_score}/10 — {vr_lbl}" if simple_mode else "VaR 95% (monthly)"
    st.metric(lbl, f"{var_95:.1%}",
              help="In the worst 1-in-20 months, you could lose at most this much.")
with c3:
    lbl = f"Worst Drop: {dd_score}/10 — {dd_lbl}" if simple_mode else "Max Drawdown"
    st.metric(lbl, f"{max_dd:.1%}",
              help="Largest peak-to-trough decline over the full period.")
with c4:
    beta_desc = ("Amplifies market" if beta > 1.2 else "Defensive" if beta < 0.8 else "Tracks market")
    lbl = f"Market Feel — {beta_desc}" if simple_mode else "Beta"
    st.metric(lbl, f"{beta:.2f}",
              help="How much your portfolio moves relative to the S&P 500.")

st.divider()

# ── Chart 1 + 2 ───────────────────────────────────────────────────────────────
col_a, col_b = st.columns([3, 2])

with col_a:
    st.subheader("Drawdown — How Far Below Peak?" if simple_mode else "Drawdown History")
    worst_date = drawdown.idxmin()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values * 100,
        fill="tozeroy", fillcolor="rgba(216,90,48,0.15)",
        line=dict(color="#D85A30", width=1.5), name="Drawdown"
    ))
    fig2.add_annotation(
        x=worst_date, y=drawdown.min() * 100,
        text=f"Worst: {drawdown.min():.1%}",
        showarrow=True, arrowhead=2, arrowcolor="#D85A30",
        font=dict(color="#D85A30"), ax=50, ay=40
    )
    fig2.update_layout(
        yaxis_title="% below peak", yaxis_ticksuffix="%",
        height=300, margin=dict(l=60, r=20, t=30, b=40)
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.subheader("Monthly Returns" if simple_mode else "Return Distribution")
    pos = (monthly > 0).sum()
    neg = (monthly <= 0).sum()
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=monthly.values * 100, nbinsx=26,
        marker_color="#7F77DD", opacity=0.75
    ))
    fig3.add_vline(
        x=var_95 * 100, line_dash="dash", line_color="#D85A30", line_width=2,
        annotation_text=f"VaR: {var_95:.1%}", annotation_position="top right",
        annotation_font_color="#D85A30"
    )
    fig3.update_layout(
        xaxis_title="Monthly return (%)", yaxis_title="Count",
        height=300, margin=dict(l=50, r=20, t=30, b=40)
    )
    st.plotly_chart(fig3, use_container_width=True)
    if simple_mode:
        st.caption(f"✅ Up months: **{pos}** &nbsp;&nbsp; ❌ Down months: **{neg}**")

st.divider()

# ── Chart 3 + 4 ───────────────────────────────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Do Your Stocks Move Together?" if simple_mode else "Correlation Heatmap")
    if simple_mode:
        st.caption("1.0 = always move together. Values closer to 0 = better diversification.")
    fig4 = go.Figure(go.Heatmap(
        z=corr.values, x=tickers, y=tickers,
        colorscale=[[0,"#378ADD"],[0.5,"#F1EFE8"],[1,"#7F77DD"]],
        zmin=-1, zmax=1,
        text=corr.round(2).values, texttemplate="%{text}", textfont_size=12
    ))
    fig4.update_layout(height=340, margin=dict(l=60, r=20, t=30, b=60))
    st.plotly_chart(fig4, use_container_width=True)

with col_d:
    st.subheader("What Industries Do You Own?" if simple_mode else "Sector Allocation")
    sector_palette = ["#7F77DD","#1D9E75","#EF9F27","#378ADD","#D85A30","#888780","#D4537E","#639922"]
    fig5 = go.Figure(go.Pie(
        labels=list(sector_wt.keys()),
        values=list(sector_wt.values()),
        hole=0.45,
        marker_colors=sector_palette[:len(sector_wt)],
        textinfo="label+percent"
    ))
    fig5.add_annotation(
        text=f"{len(sector_wt)}<br>sectors",
        x=0.5, y=0.5, font_size=14, showarrow=False
    )
    fig5.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig5, use_container_width=True)

st.divider()

# ── Chart 5: Rolling Sharpe ───────────────────────────────────────────────────
st.subheader("Was Performance Consistent?" if simple_mode else "Rolling Sharpe Ratio (63-day)")
if simple_mode:
    st.caption("Above 1.0 = good returns for the risk taken. Below 0 = risk-adjusted loss.")

fig6 = go.Figure()
fig6.add_hrect(y0=1,  y1=5,  fillcolor="rgba(29,158,117,0.07)",  line_width=0)
fig6.add_hrect(y0=-5, y1=0,  fillcolor="rgba(216,90,48,0.07)",   line_width=0)
fig6.add_trace(go.Scatter(
    x=rolling_sharpe.index, y=rolling_sharpe.values,
    line=dict(color="#7F77DD", width=1.8), name="Rolling Sharpe"
))
fig6.add_hline(y=1, line_dash="dash", line_color="#1D9E75", line_width=1.5,
               annotation_text="Good (1.0)", annotation_position="right",
               annotation_font_color="#1D9E75")
fig6.add_hline(y=0, line_color="#D85A30", line_width=0.8)
fig6.update_layout(
    yaxis_title="Sharpe ratio", height=290,
    margin=dict(l=60, r=90, t=30, b=40)
)
st.plotly_chart(fig6, use_container_width=True)

st.divider()

# ── Scorecard table ───────────────────────────────────────────────────────────
st.subheader("Full Scorecard")
rows = [
    ("Annual Return",      f"{ann_ret:.1%}",  f"{sp500_ann:.1%}",  f"{'↑ Beats' if ann_ret>sp500_ann else '↓ Trails'} S&P 500"),
    ("Annual Volatility",  f"{ann_vol:.1%}",  f"{sp500_vol:.1%}",  f"{'More volatile' if ann_vol>sp500_vol else 'Less volatile'} than market"),
    ("Sharpe Ratio",       f"{sharpe:.2f}",   f"{sp500_sharpe:.2f}", f"{sh_score}/10 — {sh_lbl}"),
    ("Max Drawdown",       f"{max_dd:.1%}",   "—",                  f"{dd_score}/10 — {dd_lbl}"),
    ("VaR 95% (monthly)",  f"{var_95:.1%}",   "—",                  f"{vr_score}/10 — {vr_lbl}"),
    ("Beta",               f"{beta:.2f}",     "1.00",               "Amplifies market" if beta>1.2 else "Defensive" if beta<0.8 else "Tracks market"),
]
df_sc = pd.DataFrame(rows, columns=["Metric","Your Portfolio","S&P 500","Plain English"])
st.dataframe(df_sc, use_container_width=True, hide_index=True)
