import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import chart, table, sidebar, need_portfolio, sharpe_label, var_label, dd_label, beta_label

st.set_page_config(page_title="Risk", page_icon="📉", layout="wide")
simple = sidebar()
d = need_portfolio()

sh_lbl, sh_sc = sharpe_label(d["sharpe"])
vr_lbl, vr_sc = var_label(d["var_95"])
dd_lbl, dd_sc = dd_label(d["max_dd"])
tickers = d["tickers"]

st.title("📉 Risk")
st.caption("How your portfolio behaved, and how much risk it took to get there.")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Quality {sh_sc}/10 · {sh_lbl}" if simple else "Sharpe ratio", f"{d['sharpe']:.2f}",
          help="Return earned per unit of risk. Above 1.0 is good.")
c2.metric(f"Bad month {vr_sc}/10 · {vr_lbl}" if simple else "VaR 95% (monthly)", f"{d['var_95']:.1%}",
          help="In the worst 1-in-20 months, the loss was about this big or more.")
c3.metric(f"Worst drop {dd_sc}/10 · {dd_lbl}" if simple else "Max drawdown", f"{d['max_dd']:.1%}",
          help="Largest fall from a peak.")
c4.metric(beta_label(d["beta"]) if simple else "Beta vs S&P 500", f"{d['beta']:.2f}",
          help="1.0 = moves with the market. Higher = bigger swings.")

st.divider()
a, b = st.columns([3, 2])
with a:
    dd = d["drawdown"]
    fig = go.Figure(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy",
                               fillcolor="rgba(216,90,48,0.15)", line=dict(color="#D85A30", width=1.5)))
    fig.add_annotation(x=dd.idxmin(), y=dd.min() * 100, text=f"Worst {dd.min():.1%}",
                       showarrow=True, arrowhead=2, ax=50, ay=30)
    fig.update_layout(title="How far below its peak" if simple else "Drawdown",
                      yaxis_ticksuffix="%", showlegend=False)
    chart(fig, height=320)
with b:
    m = d["monthly_ret"]
    fig = go.Figure(go.Histogram(x=m * 100, nbinsx=26, marker_color="#7F77DD", opacity=0.8))
    fig.add_vline(x=d["var_95"] * 100, line_dash="dash", line_color="#D85A30",
                  annotation_text=f"Bad month {d['var_95']:.1%}")
    fig.update_layout(title="Good months vs bad months" if simple else "Monthly return distribution",
                      xaxis_ticksuffix="%", showlegend=False)
    chart(fig, height=320)
    st.caption(f"✅ Up months: **{(m > 0).sum()}** · ❌ Down months: **{(m <= 0).sum()}**")

a, b = st.columns(2)
with a:
    corr = d["corr_matrix"]
    fig = go.Figure(go.Heatmap(z=corr.values, x=tickers, y=tickers, zmin=-1, zmax=1,
                               colorscale=[[0, "#378ADD"], [0.5, "#F1EFE8"], [1, "#7F77DD"]],
                               text=corr.round(2).values, texttemplate="%{text}"))
    fig.update_layout(title="Do your stocks move together?" if simple else "Correlation matrix")
    chart(fig, height=360)
    n = len(corr)
    avg = corr.values[np.triu_indices(n, k=1)].mean() if n > 1 else 1.0
    if simple:
        st.caption(f"Average correlation **{avg:.2f}** — "
                   + ("high: your stocks tend to fall together." if avg > 0.6 else
                      "moderate diversification." if avg > 0.3 else "good diversification."))
with b:
    sw = d["sector_weights"]
    fig = go.Figure(go.Pie(labels=list(sw), values=list(sw.values()), hole=0.5, textinfo="label+percent"))
    fig.update_layout(title="What industries you own" if simple else "Sector allocation", showlegend=False)
    chart(fig, height=360)
    top = max(sw, key=sw.get)
    if sw[top] > 40:
        st.caption(f"⚠️ {sw[top]:.0f}% is in **{top}** — a single-sector bet.")

rs = d["rolling_sharpe"].dropna()
fig = go.Figure(go.Scatter(x=rs.index, y=rs, line=dict(color="#7F77DD", width=1.8)))
fig.add_hline(y=1, line_dash="dash", line_color="#1D9E75", annotation_text="Good (1.0)")
fig.add_hline(y=0, line_color="#D85A30", line_width=0.8)
fig.update_layout(title="Was performance consistent?" if simple else "Rolling 3-month Sharpe ratio",
                  showlegend=False)
chart(fig, height=300)

st.subheader("Scorecard")
table(pd.DataFrame([
    ("Annual return", f"{d['ann_return']:.1%}", f"{d['sp500_ann']:.1%}",
     "Beats the S&P 500" if d["ann_return"] > d["sp500_ann"] else "Trails the S&P 500"),
    ("Annual volatility", f"{d['ann_vol']:.1%}", f"{d['sp500_vol']:.1%}",
     "More volatile than the market" if d["ann_vol"] > d["sp500_vol"] else "Calmer than the market"),
    ("Sharpe ratio", f"{d['sharpe']:.2f}", f"{d['sp500_sharpe']:.2f}", f"{sh_sc}/10 · {sh_lbl}"),
    ("Max drawdown", f"{d['max_dd']:.1%}", "—", f"{dd_sc}/10 · {dd_lbl}"),
    ("VaR 95% (monthly)", f"{d['var_95']:.1%}", "—", f"{vr_sc}/10 · {vr_lbl}"),
    ("Beta", f"{d['beta']:.2f}", "1.00", beta_label(d["beta"])),
], columns=["Metric", "Your portfolio", "S&P 500", "Plain English"]), hide_index=True)
