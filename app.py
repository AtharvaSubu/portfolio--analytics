import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_engine import load_portfolio_data
from ui import chart, sidebar, sharpe_label, var_label, dd_label, table

st.set_page_config(page_title="Portfolio Analytics", page_icon="📊", layout="wide")
simple_mode = sidebar()

DEFAULT_HOLDINGS = pd.DataFrame({
    "Ticker": ["AAPL", "MSFT", "TSLA", "JPM", "XOM"],
    "Weight %": [25.0, 20.0, 15.0, 20.0, 20.0],
})
if "holdings" not in st.session_state:
    st.session_state["holdings"] = DEFAULT_HOLDINGS

st.title("📊 Portfolio Analytics")
st.caption("Institutional-grade portfolio analysis in plain English. Live data from Yahoo Finance.")

# ── Portfolio input ───────────────────────────────────────────────────────────
with st.form("portfolio_form"):
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Your holdings** — click a cell to edit, use the last row to add a stock.")
        edited = st.data_editor(
            st.session_state["holdings"],
            num_rows="dynamic",
            hide_index=True,
            key=f"holdings_editor_{st.session_state.get('editor_v', 0)}",
            column_config={
                "Ticker": st.column_config.TextColumn(
                    "Ticker", help="Yahoo Finance symbol, e.g. AAPL, RELIANCE.NS, BP.L, SPY"),
                "Weight %": st.column_config.NumberColumn(
                    "Weight %", min_value=0.0, max_value=100.0, step=1.0, format="%.1f"),
            },
        )
    with right:
        st.markdown("**Tips**")
        st.caption(
            "🇺🇸 US: `AAPL`, `NVDA`  \n🇮🇳 India: `RELIANCE.NS`, `TCS.NS`  \n"
            "🇬🇧 UK: `BP.L`  \n📈 ETFs: `SPY`, `QQQ`"
        )
        normalise = st.checkbox("Scale weights to 100% automatically", value=True,
                                help="Enter weights in any units (e.g. dollars held) and they'll be scaled.")
    submitted = st.form_submit_button("🔍 Analyze portfolio", type="primary")


def clean_holdings(df, normalise):
    df = df.copy()
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Weight %"] = pd.to_numeric(df["Weight %"], errors="coerce")
    df = df[(df["Ticker"] != "") & (df["Ticker"] != "NONE") & (df["Ticker"] != "NAN")]
    df = df.dropna(subset=["Weight %"])
    df = df[df["Weight %"] > 0].groupby("Ticker", as_index=False, sort=False)["Weight %"].sum()

    if len(df) < 2:
        return None, "Please enter at least 2 holdings with a weight above 0."
    total = df["Weight %"].sum()
    if not normalise and abs(total - 100) > 0.5:
        return None, f"Weights add up to {total:.1f}%. Make them total 100%, or tick 'Scale weights'."
    return dict(zip(df["Ticker"], df["Weight %"] / total)), None


if submitted:
    st.session_state["holdings"] = edited.reset_index(drop=True)
    st.session_state["editor_v"] = st.session_state.get("editor_v", 0) + 1
    portfolio, err = clean_holdings(edited, normalise)
    if err:
        st.error(err)
    else:
        ok = False
        with st.spinner("Fetching live market data…"):
            try:
                st.session_state["portfolio_data"] = load_portfolio_data(tuple(portfolio.items()))
                ok = True
            except Exception as e:
                st.session_state["portfolio_data"] = None
                st.error(f"Couldn't analyze this portfolio: {e}")
        if ok:
            st.rerun()  # refresh so the sidebar shows the new portfolio

d = st.session_state.get("portfolio_data")

if d is None:
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.markdown("#### 📉 Risk\nReturns, drawdowns, worst-month risk, diversification.")
    c2.markdown("#### 🧬 Factors\nWhat's really driving your returns — size, value, profitability.")
    c3.markdown("#### 💱 FX\nGlobal currency risk today, and how the dollar affects your holdings.")
    st.stop()

# ── Results ───────────────────────────────────────────────────────────────────
if d.get("dropped"):
    st.warning(f"No price history found for {', '.join(d['dropped'])} — analyzed the rest and rescaled weights.")

st.divider()
st.caption(f"Analysis period: {d['start_date']} → {d['end_date']}")

sh_lbl, sh_sc = sharpe_label(d["sharpe"])
vr_lbl, vr_sc = var_label(d["var_95"])
dd_lbl, dd_sc = dd_label(d["max_dd"])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Yearly return" if simple_mode else "Annual return", f"{d['ann_return']:.1%}",
          f"{d['ann_return'] - d['sp500_ann']:+.1%} vs S&P 500")
c2.metric("Risk level" if simple_mode else "Volatility", f"{d['ann_vol']:.1%}",
          f"{d['ann_vol'] - d['sp500_vol']:+.1%} vs S&P 500", delta_color="inverse")
c3.metric(f"Quality ({sh_sc}/10)" if simple_mode else "Sharpe ratio", f"{d['sharpe']:.2f}",
          f"{d['sharpe'] - d['sp500_sharpe']:+.2f} vs S&P 500")
c4.metric(f"Worst drop ({dd_sc}/10)" if simple_mode else "Max drawdown", f"{d['max_dd']:.1%}")
c5.metric(f"Bad month ({vr_sc}/10)" if simple_mode else "VaR 95% (monthly)", f"{d['var_95']:.1%}")

if simple_mode:
    beats = d["ann_return"] > d["sp500_ann"]
    st.info(
        f"Your portfolio **{'beat' if beats else 'trailed'}** the S&P 500 by "
        f"**{abs(d['ann_return'] - d['sp500_ann']):.1%} a year**. Return quality is **{sh_lbl.lower()}**. "
        f"In a bad month (1 in 20), expect to lose up to **{abs(d['var_95']):.1%}**. "
        f"The worst drop from a peak was **{d['max_dd']:.1%}** ({dd_lbl.lower()})."
    )

cp, cs = d["cum_port"], d["cum_sp500"]
fig = go.Figure()
fig.add_trace(go.Scatter(x=cp.index, y=cp * 10000, name=f"Your portfolio ({cp.iloc[-1]-1:+.0%})",
                         line=dict(color="#7F77DD", width=2.5)))
fig.add_trace(go.Scatter(x=cs.index, y=cs * 10000, name=f"S&P 500 ({cs.iloc[-1]-1:+.0%})",
                         line=dict(color="#888780", width=1.5, dash="dot")))
fig.update_layout(title="Growth of $10,000", yaxis_title="Value ($)", hovermode="x unified",
                  legend=dict(orientation="h", y=1.08))
chart(fig, height=380)

with st.expander("📋 Holdings detail"):
    rows = []
    for t, w in zip(d["tickers"], d["weights"]):
        f = d["fundamentals"][t]
        rows.append({"Ticker": t, "Company": f["company"], "Sector": f["sector"], "Weight": f"{w:.1%}",
                     "Market cap": f"${f['market_cap']/1e9:,.0f}B" if f["market_cap"] else "—",
                     "P/E": f"{f['pe_ratio']:.1f}" if f["pe_ratio"] else "—"})
    table(pd.DataFrame(rows), hide_index=True)

st.caption("Explore **Risk**, **Factors** and **FX** in the sidebar.")
