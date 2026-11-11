# Portfolio Analytics

Free portfolio analysis in plain English. Enter your holdings and get risk, factor and currency analysis built on live Yahoo Finance data.

## Pages
- **Home** — enter holdings (editable table), headline scorecard, growth of $10,000
- **Risk** — drawdowns, worst-month risk, correlations, sector mix, consistency
- **Factors** — Fama-French 5-factor exposures for the portfolio and each stock
- **FX**
  - *Global currency risk* — 14 currencies vs the dollar: performance heatmap, volatility regimes, stress signals, carry vs risk, custom pair lookup. Works without a portfolio.
  - *Your portfolio's currency risk* — estimated foreign revenue plus measured dollar sensitivity (weekly regression on S&P 500 and the US Dollar Index), per-stock dollar betas and a stress test

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Files
    app.py            Home page + portfolio input
    data_engine.py    Data fetching (cached) and all calculations
    ui.py             Shared sidebar, chart helpers, plain-English scores
    pages/            Risk, Factors, FX
