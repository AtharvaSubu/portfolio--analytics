import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_engine import (
    COLORS, CURRENCIES, POLICY_RATES, POLICY_RATES_AS_OF,
    date_window, fetch_global_fx, fetch_custom_pair,
    period_changes, risk_table, regime,
    estimated_fx_exposure, dollar_sensitivity,
)
from ui import chart, table, sidebar

st.set_page_config(page_title="FX", page_icon="💱", layout="wide")
simple = sidebar()

st.title("💱 Currency Risk")

start, end = date_window()
with st.spinner("Loading global currency data…"):
    strength, dxy = fetch_global_fx(start, end)

if strength.empty:
    st.error("Currency data isn't available from Yahoo Finance right now. Please try again in a minute.")
    st.stop()

tab_global, tab_port = st.tabs(["🌍 Global currency risk", "💼 Your portfolio's currency risk"])

REGIME_COLOR = {"Calm": "#1D9E75", "Elevated": "#EF9F27", "Stormy": "#D85A30"}

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — GLOBAL (no portfolio needed)
# ═════════════════════════════════════════════════════════════════════════════
with tab_global:
    st.caption(
        "How the world's major currencies are moving against the US dollar right now. "
        "Positive numbers mean the currency got **stronger** vs the dollar."
        if simple else
        "All currencies expressed as USD per unit of foreign currency (positive = foreign currency appreciated)."
    )

    # Editable policy rates (used for carry)
    if "policy_rates" not in st.session_state:
        st.session_state["policy_rates"] = dict(POLICY_RATES)
    with st.expander(f"⚙️ Interest rates used for carry (defaults as of {POLICY_RATES_AS_OF})"):
        rates_df = pd.DataFrame({"Currency": list(st.session_state["policy_rates"]),
                                 "Policy rate %": list(st.session_state["policy_rates"].values())})
        edited_rates = st.data_editor(rates_df, hide_index=True, disabled=["Currency"], key="rates_editor")
        st.session_state["policy_rates"] = dict(zip(edited_rates["Currency"], edited_rates["Policy rate %"]))
        st.caption("Central banks change rates several times a year — edit these to keep carry numbers current.")

    rates = st.session_state["policy_rates"]
    changes = period_changes(strength)
    risk = risk_table(strength, rates)
    risk["regime"] = risk["vol_1m"].apply(regime)

    # ── Headline KPIs ─────────────────────────────────────────────────────────
    dxy_1m = dxy.iloc[-1] / dxy.loc[:dxy.index[-1] - pd.DateOffset(months=1)].iloc[-1] - 1
    dxy_1y = dxy.iloc[-1] / dxy.loc[:dxy.index[-1] - pd.DateOffset(years=1)].iloc[-1] - 1
    stormiest = risk["vol_1m"].idxmax()
    mover = changes["1M"].abs().idxmax()
    n_stressed = int((risk["stress"] > 1.3).sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("US Dollar Index", f"{dxy.iloc[-1]:.1f}", f"{dxy_1m:+.1%} past month")
    k2.metric("Most volatile right now", f"{stormiest} · {CURRENCIES[stormiest]['name']}",
              f"{risk.loc[stormiest, 'vol_1m']:.0%} annualized", delta_color="off")
    k3.metric("Biggest mover (1 month)", f"{mover}", f"{changes.loc[mover, '1M']:+.1%} vs USD")
    k4.metric("Currencies under stress", f"{n_stressed} of {len(risk)}",
              help="Stress = recent volatility at least 30% above its 1-year average.", delta_color="off")

    # Plain-English summary
    dollar_dir = "strengthened" if dxy_1m > 0 else "weakened"
    weakest = changes["1M"].idxmin()
    strongest = changes["1M"].idxmax()
    stressed = risk[risk["stress"] > 1.3].index.tolist()
    summary = (
        f"Over the past month the US dollar **{dollar_dir} {abs(dxy_1m):.1%}** "
        f"({dxy_1y:+.1%} over a year). "
        f"**{CURRENCIES[strongest]['name']}** gained the most ({changes.loc[strongest, '1M']:+.1%}) and "
        f"**{CURRENCIES[weakest]['name']}** lost the most ({changes.loc[weakest, '1M']:+.1%}). "
    )
    summary += (f"Volatility is unusually high in **{', '.join(stressed)}** — expect bigger swings there."
                if stressed else "No currency is showing unusual volatility right now.")
    st.info(summary)

    # ── Performance heatmap ───────────────────────────────────────────────────
    st.subheader("Who's winning and losing vs the dollar" if simple else "Performance vs USD")
    order = changes["1M"].sort_values(ascending=False).index
    hm = changes.loc[order]
    labels = [f"{c} · {CURRENCIES[c]['name']}" for c in order]
    lim = float(np.nanmax(np.abs(hm.values))) or 0.01
    fig = go.Figure(go.Heatmap(
        z=hm.values * 100, x=hm.columns, y=labels, zmin=-lim * 100, zmax=lim * 100,
        colorscale=[[0, "#D85A30"], [0.5, "#F1EFE8"], [1, "#1D9E75"]],
        text=[[f"{v:+.1%}" if pd.notna(v) else "" for v in row] for row in hm.values],
        texttemplate="%{text}", showscale=False,
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    chart(fig, height=40 * len(order) + 80)

    # ── Risk ranking ──────────────────────────────────────────────────────────
    a, b = st.columns(2)
    with a:
        st.subheader("Calm or stormy?" if simple else "Realized volatility")
        r = risk.sort_values("vol_1m")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=r.index, x=r["vol_1m"] * 100, orientation="h", name="Last month",
                             marker_color=[REGIME_COLOR[g] for g in r["regime"]],
                             text=[f"{v:.1%}" for v in r["vol_1m"]], textposition="outside"))
        fig.add_trace(go.Scatter(y=r.index, x=r["vol_1y"] * 100, mode="markers", name="1-year average",
                                 marker=dict(symbol="line-ns-open", size=14, color="#444", line=dict(width=2))))
        fig.update_layout(xaxis_title="Annualized volatility (%)", legend=dict(orientation="h", y=1.08))
        chart(fig, height=40 * len(r) + 80)
        st.caption("Bar = last month · tick = 1-year average. 🟢 calm <6% · 🟡 elevated 6–12% · 🔴 stormy >12%")
    with b:
        st.subheader("Does holding it pay?" if simple else "Carry vs risk")
        cr = risk.dropna(subset=["carry"])
        fig = go.Figure(go.Scatter(
            x=cr["vol_1y"] * 100, y=cr["carry"] * 100, mode="markers+text", text=cr.index,
            textposition="top center",
            marker=dict(size=12, color=["#1D9E75" if c > 0 else "#D85A30" for c in cr["carry"]]),
            hovertemplate="%{text}<br>Vol %{x:.1f}%<br>Carry %{y:+.2f}%<extra></extra>",
        ))
        fig.add_hline(y=0, line_color="#888780", line_dash="dash")
        fig.update_layout(xaxis_title="Risk: 1-year volatility (%)",
                          yaxis_title="Carry: rate vs USD (%)", showlegend=False)
        chart(fig, height=40 * len(r) + 80)
        best = cr["carry_to_risk"].idxmax() if len(cr) else None
        if best:
            st.caption(f"Carry = interest you earn (or pay) holding the currency instead of dollars. "
                       f"Best carry per unit of risk: **{best}**. High carry usually comes with crash risk.")

    # ── Price history for chosen currencies ───────────────────────────────────
    st.subheader("Trend over time")
    c1, c2 = st.columns([3, 1])
    with c1:
        picks = st.multiselect("Currencies", list(strength.columns),
                               default=[c for c in ["EUR", "JPY", "GBP", "INR"] if c in strength.columns],
                               format_func=lambda c: f"{c} · {CURRENCIES[c]['name']}")
    with c2:
        span = st.radio("Period", ["1Y", "3Y", "5Y"], horizontal=True, index=0)
    if picks:
        years = int(span[0])
        sub = strength[picks].loc[strength.index[-1] - pd.DateOffset(years=years):].dropna(how="all")
        norm = sub / sub.bfill().iloc[0] * 100
        fig = go.Figure()
        for i, c in enumerate(picks):
            fig.add_trace(go.Scatter(x=norm.index, y=norm[c], name=c, line=dict(color=COLORS[i % len(COLORS)])))
        fig.add_hline(y=100, line_color="#888780", line_dash="dot")
        fig.update_layout(yaxis_title="Value vs USD (start = 100)", hovermode="x unified",
                          legend=dict(orientation="h", y=1.08))
        chart(fig, height=360)

        if len(picks) > 1:
            rets_1y = strength[picks].pct_change().loc[strength.index[-1] - pd.DateOffset(years=1):]
            corr = rets_1y.corr()
            fig = go.Figure(go.Heatmap(z=corr.values, x=picks, y=picks, zmin=-1, zmax=1,
                                       colorscale=[[0, "#378ADD"], [0.5, "#F1EFE8"], [1, "#7F77DD"]],
                                       text=corr.round(2).values, texttemplate="%{text}"))
            fig.update_layout(title="Do they move together? (1-year daily correlation)")
            chart(fig, height=320)

    # ── Scorecard ─────────────────────────────────────────────────────────────
    st.subheader("Currency scorecard")
    sc = pd.DataFrame({
        "Currency": [f"{c} · {CURRENCIES[c]['name']}" for c in risk.index],
        "Region": risk["region"],
        "1M vs USD": changes.loc[risk.index, "1M"],
        "1Y vs USD": changes.loc[risk.index, "1Y"],
        "Vol (1M)": risk["vol_1m"],
        "Regime": risk["regime"],
        "Off 1Y high": risk["off_peak"],
        "Bad month (VaR 95%)": risk["var95_month"],
        "Carry vs USD": risk["carry"],
    }).sort_values("Vol (1M)", ascending=False)
    table(sc.style.format({
        "1M vs USD": "{:+.1%}", "1Y vs USD": "{:+.1%}", "Vol (1M)": "{:.1%}",
        "Off 1Y high": "{:.1%}", "Bad month (VaR 95%)": "{:.1%}", "Carry vs USD": "{:+.2%}",
    }), hide_index=True)

    # ── Custom pair ───────────────────────────────────────────────────────────
    with st.expander("➕ Look up any other currency pair"):
        st.caption("Yahoo Finance format `XXXYYY=X`, e.g. `GBPJPY=X`, `EURCHF=X`, `USDTHB=X`, `USDIDR=X`.")
        sym = st.text_input("Ticker", placeholder="GBPJPY=X").strip().upper()
        if sym:
            s = fetch_custom_pair(sym, start, end)
            if s is None:
                st.warning(f"No data found for `{sym}`.")
            else:
                r = s.pct_change().dropna()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Latest", f"{s.iloc[-1]:,.4f}")
                m2.metric("1Y change", f"{s.iloc[-1] / s.loc[:s.index[-1] - pd.DateOffset(years=1)].iloc[-1] - 1:+.1%}")
                v = r.tail(21).std() * np.sqrt(252)
                m3.metric("Volatility (1M)", f"{v:.1%}", regime(v), delta_color="off")
                m4.metric("Worst day (5Y)", f"{r.min():.1%}")
                fig = go.Figure(go.Scatter(x=s.index, y=s, line=dict(color="#7F77DD")))
                fig.update_layout(title=sym, showlegend=False)
                chart(fig, height=300)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — YOUR PORTFOLIO
# ═════════════════════════════════════════════════════════════════════════════
with tab_port:
    d = st.session_state.get("portfolio_data")
    if d is None:
        st.info("👈 Go to **Home**, enter your holdings and click **Analyze portfolio** to see "
                "how currencies affect your portfolio.")
    else:
        st.caption(
            "Even US stocks earn money abroad, so a stronger dollar can quietly hurt them. "
            "This tab shows that risk two ways: an **estimate** from where companies earn revenue, "
            "and a **measurement** of how your portfolio actually reacted to the dollar."
        )

        sens = dollar_sensitivity(d["port_ret"], d["sp500_ret"], d["returns"], dxy)
        exposure = estimated_fx_exposure(d["tickers"], d["weights"], d["fundamentals"])
        foreign = sum(v for k, v in exposure.items() if k != "USD")

        k1, k2, k3 = st.columns(3)
        k1.metric("Estimated foreign revenue", f"{foreign:.0f}%",
                  help="Rough estimate from each holding's sector — not company-reported figures.")
        if sens:
            b = sens["usd_beta"]
            k2.metric("Measured dollar sensitivity", f"{b:+.2f}",
                      help="Weekly regression on S&P 500 + US Dollar Index. "
                           "Portfolio % move per 1% dollar move, beyond what the market explains.")
            label = ("hurt by a strong dollar" if b < -0.1 else
                     "helped by a strong dollar" if b > 0.1 else "barely affected by the dollar")
            k3.metric("In plain English", label.capitalize(), delta_color="off")

            st.info(
                f"Historically, when the dollar rose **1%** in a week, your portfolio moved "
                f"**{b:+.2f}%** beyond what the stock market alone would explain. "
                + ("That means a strong-dollar period is a real headwind for you."
                   if b < -0.1 else
                   "So a stronger dollar has tended to help you — common for importers and domestic firms."
                   if b > 0.1 else
                   "Currency moves haven't mattered much for this portfolio.")
                + f" (Based on {sens['weeks']} weeks of data.)"
            )
        else:
            k2.metric("Measured dollar sensitivity", "—")
            st.warning("Not enough overlapping data to measure dollar sensitivity.")

        a, b_col = st.columns(2)
        with a:
            st.subheader("Where your revenue comes from (estimate)")
            fig = go.Figure(go.Bar(
                x=list(exposure), y=list(exposure.values()),
                marker_color=["#B4B2A9" if c == "USD" else "#7F77DD" for c in exposure],
                text=[f"{v:.0f}%" for v in exposure.values()], textposition="outside",
            ))
            fig.update_layout(yaxis_title="% of portfolio", showlegend=False)
            chart(fig, height=340)
            st.caption("Based on typical revenue mix for each holding's sector. "
                       "Company 10-K geographic segments would be more precise.")
        with b_col:
            st.subheader("Which holdings react most to the dollar")
            if sens:
                ps = sens["per_stock"]
                fig = go.Figure(go.Bar(
                    y=ps.index, x=ps.values, orientation="h",
                    marker_color=["#D85A30" if v < 0 else "#1D9E75" for v in ps.values],
                    text=[f"{v:+.2f}" for v in ps.values], textposition="outside",
                ))
                fig.add_vline(x=0, line_color="#888780")
                fig.update_layout(xaxis_title="% move per 1% dollar move", showlegend=False)
                chart(fig, height=340)
                st.caption("Red = falls when the dollar strengthens. Green = rises.")

        st.subheader("Dollar stress test")
        shocks = [-0.10, -0.05, 0.05, 0.10]
        labels = ["Dollar −10%", "Dollar −5%", "Dollar +5%", "Dollar +10%"]
        est = [-s * foreign for s in shocks]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=est, name="Estimate (revenue exposure)", marker_color="#B4B2A9",
                             text=[f"{v:+.1f}%" for v in est], textposition="outside"))
        if sens:
            meas = [s * 100 * sens["usd_beta"] for s in shocks]
            fig.add_trace(go.Bar(x=labels, y=meas, name="Based on measured sensitivity",
                                 marker_color="#7F77DD",
                                 text=[f"{v:+.1f}%" for v in meas], textposition="outside"))
        fig.add_hline(y=0, line_color="#888780")
        fig.update_layout(barmode="group", yaxis_title="Portfolio impact (%)",
                          legend=dict(orientation="h", y=1.1))
        chart(fig, height=360)
        st.caption("The estimate assumes foreign revenue translates one-for-one; the measured bars use how "
                   "your holdings actually behaved. Real outcomes depend on hedging and pricing power.")

        if sens and len(sens["rolling_corr"]):
            rc = sens["rolling_corr"]
            fig = go.Figure(go.Scatter(x=rc.index, y=rc, line=dict(color="#7F77DD"), fill="tozeroy",
                                       fillcolor="rgba(127,119,221,0.12)"))
            fig.add_hline(y=0, line_color="#888780")
            fig.update_layout(title="Is the dollar link getting stronger? (52-week rolling correlation)",
                              yaxis_range=[-1, 1], showlegend=False)
            chart(fig, height=280)
