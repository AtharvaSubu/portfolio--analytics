"""Shared sidebar, chart helpers and plain-English scoring used by every page."""

import streamlit as st


def chart(fig, height=None):
    """Full-width Plotly chart that works on both old and new Streamlit versions."""
    if height:
        fig.update_layout(height=height)
    fig.update_layout(margin=dict(l=50, r=20, t=40, b=40))
    try:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def table(df, **kwargs):
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


def sidebar():
    """Display-mode toggle + current portfolio, identical on every page."""
    with st.sidebar:
        simple_now = st.session_state.get("simple_mode", True)
        choice = st.radio(
            "Display mode",
            ["Simple — plain English", "Advanced — raw numbers"],
            index=0 if simple_now else 1,
        )
        st.session_state["simple_mode"] = choice.startswith("Simple")

        d = st.session_state.get("portfolio_data")
        if d:
            st.divider()
            st.caption("Current portfolio")
            for t, w in zip(d["tickers"], d["weights"]):
                st.markdown(f"**{t}** · {w:.0%}")
            st.caption("Change holdings on the Home page.")
    return st.session_state["simple_mode"]


def need_portfolio():
    d = st.session_state.get("portfolio_data")
    if d is None:
        st.info("👈 Go to **Home**, enter your holdings and click **Analyze portfolio** first.")
        st.stop()
    return d


# ── Plain-English scores ──────────────────────────────────────────────────────
def sharpe_label(s):
    for cut, lbl, sc in [(2, "Exceptional", 10), (1.5, "Excellent", 9), (1, "Good", 7),
                         (0.5, "Fair", 5), (0, "Poor", 3)]:
        if s > cut:
            return lbl, sc
    return "Very poor", 1


def var_label(v):
    p = abs(v)
    for cut, lbl, sc in [(0.03, "Very low risk", 10), (0.05, "Low risk", 8),
                         (0.08, "Moderate risk", 6), (0.12, "High risk", 4)]:
        if p < cut:
            return lbl, sc
    return "Very high risk", 2


def dd_label(dd):
    p = abs(dd)
    for cut, lbl, sc in [(0.10, "Minimal", 10), (0.20, "Manageable", 8),
                         (0.35, "Significant", 5), (0.50, "Severe", 3)]:
        if p < cut:
            return lbl, sc
    return "Extreme", 1


def beta_label(b):
    if b > 1.2:
        return "Amplifies market moves"
    if b < 0.8:
        return "Defensive vs the market"
    return "Moves with the market"
