import plotly.graph_objects as go
import streamlit as st

from ui import chart, table, sidebar, need_portfolio

st.set_page_config(page_title="Factors", page_icon="🧬", layout="wide")
simple = sidebar()
d = need_portfolio()

NAMES = {"Mkt-RF": "Market risk", "SMB": "Small vs large", "HML": "Value vs growth",
         "RMW": "Profitability", "CMA": "Investment style"}
DESC = {
    "Mkt-RF": "How much you amplify or dampen overall market moves.",
    "SMB": "Positive = leans to small companies. Negative = large companies.",
    "HML": "Positive = cheaper 'value' stocks. Negative = pricier 'growth' stocks.",
    "RMW": "Positive = highly profitable companies. Negative = thin margins.",
    "CMA": "Positive = conservative spenders. Negative = aggressive reinvestors.",
}
betas, alpha, r2 = d["factor_betas"], d["alpha"], d["r_squared"]

st.title("🧬 Factors")
st.caption("Why your portfolio performs the way it does — the underlying bets, not just the outcome."
           if simple else "OLS regression of daily portfolio excess returns on factor returns.")
if "Market-only" in d["ff_source"]:
    st.warning(f"{d['ff_source']}. Only market exposure is shown.")
else:
    st.caption(f"Source: {d['ff_source']}")

c1, c2, c3 = st.columns(3)
c1.metric("How much the factors explain" if simple else "R²", f"{r2:.0%}")
c2.metric("Extra return beyond factors" if simple else "Alpha (annualized)", f"{alpha:+.2%}/yr")
dom = max(betas, key=lambda k: abs(betas[k]) if k != "Mkt-RF" else 0) if len(betas) > 1 else "Mkt-RF"
c3.metric("Biggest style tilt" if simple else "Largest non-market loading", NAMES[dom])

factors = list(betas)
fig = go.Figure(go.Bar(
    x=[NAMES[f] if simple else f for f in factors], y=[betas[f] for f in factors],
    marker_color=["#1D9E75" if betas[f] >= 0 else "#D85A30" for f in factors],
    text=[f"{betas[f]:+.2f}" for f in factors], textposition="outside",
    hovertext=[DESC[f] for f in factors],
))
fig.add_hline(y=0, line_color="#888780")
fig.update_layout(title="What's driving your returns?" if simple else "Factor loadings", showlegend=False)
chart(fig, height=380)

if simple:
    cols = st.columns(len(factors))
    for col, f in zip(cols, factors):
        b = betas[f]
        col.markdown(f"**{NAMES[f]}**  \n{'🟢' if abs(b) > 0.3 else '🟡' if abs(b) > 0.1 else '⚪'} {b:+.2f}")
        col.caption(DESC[f])

st.divider()
st.subheader("Which stock drives what?" if simple else "Per-stock factor loadings")
sf = d["stock_factors"].copy()
if simple:
    sf = sf.rename(columns=NAMES)
fmt = {c: "{:+.2f}" for c in sf.columns}
fmt["Alpha (ann.)"] = "{:+.1%}"
fmt["R²"] = "{:.0%}"
table(sf.style.format(fmt))

rr = d["rolling_r2"]
if len(rr):
    fig = go.Figure(go.Scatter(x=rr.index, y=rr * 100, fill="tozeroy",
                               fillcolor="rgba(127,119,221,0.12)", line=dict(color="#7F77DD")))
    fig.update_layout(title="How consistent is the pattern? (6-month rolling R²)",
                      yaxis_ticksuffix="%", showlegend=False)
    chart(fig, height=280)
