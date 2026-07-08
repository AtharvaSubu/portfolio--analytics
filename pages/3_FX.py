import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_engine import COLORS, FX_PAIRS, CARRY_RATES

st.set_page_config(page_title="FX Analysis", page_icon="💱", layout="wide")

if "portfolio_data" not in st.session_state or st.session_state["portfolio_data"] is None:
    st.warning("No portfolio loaded. Go to the Home page and click **Analyze Portfolio** first.")
    st.stop()

d           = st.session_state["portfolio_data"]
simple_mode = st.session_state.get("simple_mode", True)

DAYS        = 252
fx_prices   = d["fx_prices"]
fx_returns  = d["fx_returns"]
fx_metrics  = d["fx_metrics"]
fx_corr     = d["fx_corr"]
portfolio_fx= d["portfolio_fx"]
PORTFOLIO   = d["portfolio"]
tickers     = d["tickers"]
fundamentals= d["fundamentals"]

st.title("💱 FX Analysis")
st.divider()

tab_a, tab_b = st.tabs([
    "🌍 Hidden FX Exposure  (your equity portfolio)",
    "📈 FX Trading  (active currency analysis)"
])

# ══════════════════════════════════════════════════════════════
# TAB A
# ══════════════════════════════════════════════════════════════
with tab_a:
    st.subheader("Hidden Currency Risk in Your Equity Portfolio")
    st.caption(
        "Even an all-US portfolio carries foreign currency risk — your companies earn revenue abroad. "
        "When the dollar strengthens, those earnings are worth less."
        if simple_mode else
        "Portfolio-weighted FX exposure estimated from sector-level geographic revenue defaults."
    )
    st.divider()

    non_usd = sum(v for k, v in portfolio_fx.items() if k != "USD")
    usd_pct = portfolio_fx.get("USD", 0)
    top_fx  = max((k for k in portfolio_fx if k != "USD"), key=lambda k: portfolio_fx[k], default="EUR")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Foreign Currency Exposure", f"{non_usd:.0f}%",
                  help="% of portfolio revenue earned outside the US")
    with c2:
        st.metric("USD Exposure", f"{usd_pct:.0f}%")
    with c3:
        st.metric("Largest Foreign Currency", f"{top_fx}  {portfolio_fx.get(top_fx,0):.0f}%")

    st.divider()

    bar_colors_map = {
        "USD":"#B4B2A9","EUR":"#7F77DD","GBP":"#1D9E75",
        "JPY":"#D85A30","AUD":"#EF9F27","CAD":"#378ADD",
        "INR":"#D4537E","CHF":"#639922"
    }

    col_l, col_r = st.columns([3,2])

    with col_l:
        st.subheader("Currency Exposure Breakdown")
        fig_a1 = go.Figure(go.Bar(
            x=list(portfolio_fx.keys()),
            y=list(portfolio_fx.values()),
            marker_color=[bar_colors_map.get(c, "#888780") for c in portfolio_fx],
            text=[f"{v:.1f}%" for v in portfolio_fx.values()],
            textposition="outside"
        ))
        fig_a1.update_layout(
            yaxis_title="Exposure (%)",
            height=360, margin=dict(l=60, r=20, t=30, b=60)
        )
        st.plotly_chart(fig_a1, use_container_width=True)
        if simple_mode:
            st.info(
                f"**{non_usd:.0f}%** of your equity value is effectively in foreign currencies "
                f"through international revenues. The biggest exposure is **{top_fx}** at "
                f"**{portfolio_fx.get(top_fx,0):.0f}%**."
            )

    with col_r:
        st.subheader("USD Stress Test")
        st.caption("Estimated portfolio impact if the US Dollar moves by these amounts.")

        usd_scenarios   = [-0.10, -0.05, 0, +0.05, +0.10]
        scenario_labels = ["-10%", "-5%", "No change", "+5%", "+10%"]
        impacts = [
            sum(-chg * pct / 100 for curr, pct in portfolio_fx.items() if curr != "USD") * 100
            for chg in usd_scenarios
        ]

        fig_a2 = go.Figure(go.Bar(
            x=scenario_labels, y=impacts,
            marker_color=["#1D9E75" if i >= 0 else "#D85A30" for i in impacts],
            text=[f"{v:+.1f}%" for v in impacts], textposition="outside"
        ))
        fig_a2.add_hline(y=0, line_color="#888780", line_width=1)
        fig_a2.update_layout(
            xaxis_title="USD change", yaxis_title="Portfolio impact (%)",
            height=360, margin=dict(l=60, r=20, t=30, b=60)
        )
        st.plotly_chart(fig_a2, use_container_width=True)
        if simple_mode:
            st.caption(
                f"A 10% dollar strengthening costs your portfolio "
                f"approx **{impacts[-1]:.1f}%** from FX translation alone."
            )

# ══════════════════════════════════════════════════════════════
# TAB B
# ══════════════════════════════════════════════════════════════
with tab_b:
    st.subheader("FX Trading Analysis")
    st.caption(
        "Analyze currency pairs: risk metrics, carry trade scores, volatility, and position sizing."
        if simple_mode else
        "Full FX analytics: risk, carry, correlation, volatility regimes, VaR-based position sizing."
    )
    st.divider()

    if fx_metrics.empty:
        st.warning("FX data could not be loaded. Check your internet connection.")
        st.stop()

    # Pair selector
    available_pairs = {FX_PAIRS[t][0]: t for t in fx_metrics.index if t in FX_PAIRS}
    selected_names  = st.multiselect(
        "Select pairs to analyze",
        options=list(available_pairs.keys()),
        default=list(available_pairs.keys())[:4]
    )
    selected_tickers = [available_pairs[n] for n in selected_names] if selected_names else list(fx_metrics.index)[:4]

    if not selected_tickers:
        st.info("Select at least one pair above.")
        st.stop()

    st.divider()

    # Rate history
    st.subheader("Rate History (normalized)" if not simple_mode else "How Each Currency Has Moved")
    if simple_mode:
        st.caption("Starting at 1.0. Rising = base currency got stronger. Falling = weaker.")

    n = len(selected_tickers)
    cols_r = min(4, n)
    rows_r = (n + cols_r - 1) // cols_r
    fig_b1 = make_subplots(
        rows=rows_r, cols=cols_r,
        subplot_titles=[FX_PAIRS[t][0] for t in selected_tickers if t in FX_PAIRS],
        vertical_spacing=0.18, horizontal_spacing=0.07
    )
    for i, ticker in enumerate(selected_tickers):
        if ticker not in fx_prices.columns: continue
        row, col = divmod(i, cols_r)
        norm = fx_prices[ticker] / fx_prices[ticker].iloc[0]
        fig_b1.add_trace(
            go.Scatter(x=fx_prices.index, y=norm,
                       line=dict(color=COLORS[i % len(COLORS)], width=1.5),
                       showlegend=False),
            row=row+1, col=col+1
        )
    fig_b1.update_layout(height=220*rows_r, margin=dict(l=40,r=20,t=60,b=40))
    st.plotly_chart(fig_b1, use_container_width=True)

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Carry Trade Score" if not simple_mode else "Which Pairs Pay You to Hold?")
        if simple_mode:
            st.caption("Green = you earn this % per year from the interest rate difference alone.")

        fm = fx_metrics.loc[selected_tickers].sort_values("carry", ascending=False)
        fig_b2 = go.Figure(go.Bar(
            x=fm["name"], y=fm["carry"] * 100,
            marker_color=["#1D9E75" if v > 0 else "#D85A30" for v in fm["carry"]],
            text=[f"{v:+.2f}%" for v in fm["carry"] * 100],
            textposition="outside"
        ))
        fig_b2.add_hline(y=0, line_color="#888780", line_width=1)
        fig_b2.update_layout(
            yaxis_title="Annual carry (%)",
            height=340, margin=dict(l=60, r=20, t=30, b=80)
        )
        st.plotly_chart(fig_b2, use_container_width=True)

    with col_r:
        st.subheader("Risk vs Return" if not simple_mode else "Best Risk/Return Pairs?")
        if simple_mode:
            st.caption("Top-left = best. High return, low volatility. Bubble = carry strength.")

        fig_b3 = go.Figure()
        for i, ticker in enumerate(selected_tickers):
            if ticker not in fx_metrics.index: continue
            row = fx_metrics.loc[ticker]
            fig_b3.add_trace(go.Scatter(
                x=[row["ann_vol"]*100], y=[row["ann_ret"]*100],
                mode="markers+text", text=[row["name"]], textposition="top center",
                marker=dict(
                    size=max(10, abs(row["carry"]) * 500),
                    color=COLORS[i % len(COLORS)], opacity=0.8,
                    line=dict(color="#1D9E75" if row["carry"]>0 else "#D85A30", width=2)
                ),
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['name']}</b><br>"
                    f"Return: {row['ann_ret']:.1%}<br>"
                    f"Vol: {row['ann_vol']:.1%}<br>"
                    f"Sharpe: {row['sharpe']:.2f}<br>"
                    f"Carry: {row['carry']:+.2%}<extra></extra>"
                )
            ))
        fig_b3.add_hline(y=0, line_dash="dash", line_color="#888780", line_width=1)
        fig_b3.update_layout(
            xaxis_title="Volatility (%) — lower is calmer",
            yaxis_title="Return (%) — higher is better",
            height=340, margin=dict(l=60, r=20, t=30, b=60)
        )
        st.plotly_chart(fig_b3, use_container_width=True)

    st.divider()

    # Volatility regime
    st.subheader("Volatility Regime" if not simple_mode else "Calm or Stormy Right Now?")
    if simple_mode:
        st.caption("Spikes = macro events. High = risky time to trade. Low = calm market.")

    WINDOW = 21
    fig_b4 = go.Figure()
    fig_b4.add_hrect(y0=0,  y1=5,  fillcolor="rgba(29,158,117,0.06)", line_width=0)
    fig_b4.add_hrect(y0=5,  y1=10, fillcolor="rgba(239,159,39,0.06)", line_width=0)
    fig_b4.add_hrect(y0=10, y1=40, fillcolor="rgba(216,90,48,0.06)",  line_width=0)
    for i, ticker in enumerate(selected_tickers[:4]):
        if ticker not in fx_returns.columns: continue
        rv = fx_returns[ticker].rolling(WINDOW).std() * np.sqrt(DAYS) * 100
        fig_b4.add_trace(go.Scatter(
            x=rv.index, y=rv,
            name=FX_PAIRS[ticker][0] if ticker in FX_PAIRS else ticker,
            line=dict(color=COLORS[i % len(COLORS)], width=1.5)
        ))
    fig_b4.update_layout(
        yaxis_title="Annualized vol (%)",
        height=300, margin=dict(l=60, r=20, t=30, b=40)
    )
    st.plotly_chart(fig_b4, use_container_width=True)

    regime_cols = st.columns(min(4, len(selected_tickers)))
    for i, ticker in enumerate(selected_tickers[:4]):
        if ticker not in fx_returns.columns: continue
        cv = fx_returns[ticker].rolling(21).std().iloc[-1] * np.sqrt(DAYS) * 100
        regime = "🟢 Calm" if cv < 5 else ("🟡 Elevated" if cv < 10 else "🔴 Stormy")
        name   = FX_PAIRS[ticker][0] if ticker in FX_PAIRS else ticker
        with regime_cols[i]:
            st.metric(name, regime, f"{cv:.1f}% vol")

    st.divider()

    # Position sizing
    st.subheader("Position Sizing Calculator")
    st.caption("How large a position can you safely take?")

    ps1, ps2 = st.columns(2)
    with ps1:
        account = st.number_input("Account size ($)", min_value=1000, value=10000, step=1000)
    with ps2:
        risk_pct = st.slider("Max loss per trade (%)", 1, 10, 2) / 100

    sizing = []
    for ticker in selected_tickers:
        if ticker not in fx_metrics.index: continue
        row  = fx_metrics.loc[ticker]
        varp = abs(row["var95"])
        maxp = min((account * risk_pct) / varp if varp > 0 else 0, account)
        rl   = "Low" if varp < 0.03 else "Moderate" if varp < 0.06 else "High" if varp < 0.10 else "Very High"
        sizing.append({
            "Pair":         row["name"],
            "Monthly VaR":  f"{varp:.1%}",
            "Max Position": f"${maxp:,.0f}",
            "Risk Level":   rl,
            "Annual Carry": f"{row['carry']:+.2%}",
        })
    if sizing:
        st.dataframe(pd.DataFrame(sizing), use_container_width=True, hide_index=True)
        if simple_mode:
            st.info(
                f"**Max Position** = largest trade where your worst 1-in-20 month stays "
                f"within your **{risk_pct:.0%}** risk budget on a **${account:,.0f}** account."
            )
