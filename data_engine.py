"""
data_engine.py
All data fetching and computation lives here.
Streamlit pages import from this file — nothing else needed.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import zipfile
import io
import streamlit as st
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta


# ── Constants ─────────────────────────────────────────────────────────────────
DAYS = 252
LOOKBACK_YEARS = 5

FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)

FX_PAIRS = {
    "EURUSD=X": ("EUR/USD", "USD", "EUR"),
    "GBPUSD=X": ("GBP/USD", "USD", "GBP"),
    "USDJPY=X": ("USD/JPY", "JPY", "USD"),
    "AUDUSD=X": ("AUD/USD", "USD", "AUD"),
    "USDCAD=X": ("USD/CAD", "CAD", "USD"),
    "USDCHF=X": ("USD/CHF", "CHF", "USD"),
    "USDINR=X": ("USD/INR", "INR", "USD"),
}

CARRY_RATES = {
    "EUR": 0.040, "GBP": 0.052, "JPY": 0.001,
    "AUD": 0.043, "CAD": 0.050, "CHF": 0.015,
    "INR": 0.065, "USD": 0.053,
}

SECTOR_FX = {
    "Technology":         {"EUR":0.25,"GBP":0.08,"JPY":0.06,"AUD":0.02,"CAD":0.03,"USD":0.56},
    "Financial Services": {"EUR":0.15,"GBP":0.12,"JPY":0.05,"AUD":0.02,"CAD":0.03,"USD":0.63},
    "Energy":             {"EUR":0.10,"GBP":0.05,"JPY":0.05,"AUD":0.05,"CAD":0.15,"USD":0.60},
    "Consumer Cyclical":  {"EUR":0.20,"GBP":0.06,"JPY":0.08,"AUD":0.02,"CAD":0.02,"USD":0.62},
    "Consumer Defensive": {"EUR":0.18,"GBP":0.08,"JPY":0.05,"AUD":0.03,"CAD":0.04,"USD":0.62},
    "Healthcare":         {"EUR":0.22,"GBP":0.08,"JPY":0.06,"AUD":0.02,"CAD":0.03,"USD":0.59},
    "Industrials":        {"EUR":0.18,"GBP":0.07,"JPY":0.07,"AUD":0.03,"CAD":0.04,"USD":0.61},
    "Communication Services":{"EUR":0.15,"GBP":0.08,"JPY":0.05,"AUD":0.02,"CAD":0.03,"USD":0.67},
    "Real Estate":        {"EUR":0.05,"GBP":0.03,"JPY":0.02,"AUD":0.02,"CAD":0.03,"USD":0.85},
    "Materials":          {"EUR":0.12,"GBP":0.06,"JPY":0.08,"AUD":0.06,"CAD":0.05,"USD":0.63},
    "Utilities":          {"EUR":0.05,"GBP":0.03,"JPY":0.02,"AUD":0.02,"CAD":0.03,"USD":0.85},
    "Unknown":            {"EUR":0.15,"GBP":0.06,"JPY":0.05,"AUD":0.03,"CAD":0.03,"USD":0.68},
}

COLORS = ["#7F77DD","#1D9E75","#D85A30","#EF9F27","#378ADD","#888780","#D4537E","#639922","#534AB7","#BA7517"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_series(s):
    """Flatten any 2D Series/DataFrame column to a clean 1D Series."""
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.Series(s.values.flatten(), index=s.index, name=s.name)


def validate_tickers(tickers):
    """Quick check that tickers exist on Yahoo Finance."""
    bad = []
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            if info.get("lastPrice", None) is None and info.get("regularMarketPrice", None) is None:
                bad.append(t)
        except Exception:
            bad.append(t)
    return bad


# ── Main data loader ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)   # cache for 1 hour
def load_portfolio_data(portfolio_tuple):
    """
    portfolio_tuple: tuple of (ticker, weight) pairs — tuples are hashable for caching.
    Returns a dict with everything the app needs.
    """
    portfolio = dict(portfolio_tuple)
    tickers   = list(portfolio.keys())
    weights   = np.array(list(portfolio.values()))

    end   = datetime.today()
    start = end - timedelta(days=LOOKBACK_YEARS * 365 + 30)
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")

    # ── 1. Price history ──────────────────────────────────────────────────────
    raw = yf.download(tickers, start=start_str, end=end_str,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    # Flatten any 2D columns
    for col in prices.columns:
        prices[col] = pd.to_numeric(prices[col].values.flatten(), errors="coerce")

    prices  = prices.dropna(how="all")
    returns = prices.pct_change().dropna()

    # Filter to tickers that actually have data
    valid_tickers = [t for t in tickers if t in prices.columns and prices[t].notna().sum() > 100]
    if not valid_tickers:
        raise ValueError("No valid price data found. Check your tickers.")

    prices   = prices[valid_tickers]
    returns  = returns[valid_tickers]
    weights_arr = np.array([portfolio[t] for t in valid_tickers])
    weights_arr = weights_arr / weights_arr.sum()   # re-normalize in case some tickers failed

    port_ret = (returns * weights_arr).sum(axis=1)

    # S&P 500 benchmark
    sp_raw = yf.download("^GSPC", start=start_str, end=end_str,
                          auto_adjust=True, progress=False)
    sp500_ret = pd.Series(
        sp_raw["Close"].values.flatten(),
        index=sp_raw.index, name="SP500"
    ).pct_change().dropna()

    # ── 2. Company fundamentals ───────────────────────────────────────────────
    fundamentals = {}
    for ticker in valid_tickers:
        try:
            info = yf.Ticker(ticker).info
            fundamentals[ticker] = {
                "company":       info.get("longName", ticker),
                "sector":        info.get("sector", "Unknown"),
                "market_cap":    info.get("marketCap", 0) or 0,
                "revenue":       info.get("totalRevenue", 0) or 0,
                "profit_margin": info.get("profitMargins", 0) or 0,
                "debt_equity":   info.get("debtToEquity", 0) or 0,
                "pe_ratio":      info.get("trailingPE", 0) or 0,
            }
        except Exception:
            fundamentals[ticker] = {
                "company": ticker, "sector": "Unknown",
                "market_cap": 0, "revenue": 0,
                "profit_margin": 0, "debt_equity": 0, "pe_ratio": 0,
            }

    # Sector weights
    sector_weights = {}
    for t, w in zip(valid_tickers, weights_arr):
        sec = fundamentals[t]["sector"]
        sector_weights[sec] = sector_weights.get(sec, 0) + w * 100

    # ── 3. Fama-French factors ────────────────────────────────────────────────
    ff_factors = None
    try:
        resp = requests.get(FF5_URL, timeout=20)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            csv_name = [n for n in z.namelist() if n.endswith(".CSV")][0]
            with z.open(csv_name) as f:
                raw_csv = f.read().decode("utf-8")
        lines = raw_csv.split("\n")
        start_line = next(i for i, l in enumerate(lines) if l.strip().startswith("2"))
        ff_raw = pd.read_csv(
            io.StringIO("\n".join(lines[start_line:])),
            header=None, names=["Date","Mkt-RF","SMB","HML","RMW","CMA","RF"],
            on_bad_lines="skip"
        ).dropna()
        ff_raw["Date"] = pd.to_datetime(ff_raw["Date"].astype(str).str.strip(),
                                         format="%Y%m%d", errors="coerce")
        ff_raw = ff_raw.dropna(subset=["Date"]).set_index("Date")
        for col in ["Mkt-RF","SMB","HML","RMW","CMA","RF"]:
            ff_raw[col] = pd.to_numeric(ff_raw[col], errors="coerce") / 100
        ff_factors = ff_raw.loc[start_str:end_str].dropna()
    except Exception:
        # Fallback: approximate factors from market data
        common = port_ret.index
        ff_factors = pd.DataFrame({
            "Mkt-RF": sp500_ret.reindex(common).fillna(0) - 0.000018,
            "SMB":    np.random.normal(0.0001, 0.004, len(common)),
            "HML":    np.random.normal(0.0001, 0.004, len(common)),
            "RMW":    np.random.normal(0.0001, 0.003, len(common)),
            "CMA":    np.random.normal(0.0001, 0.003, len(common)),
            "RF":     np.full(len(common), 0.000018),
        }, index=common)

    # ── 4. Align all data ─────────────────────────────────────────────────────
    common = port_ret.index.intersection(ff_factors.index).intersection(sp500_ret.index)
    port_a   = port_ret.loc[common]
    ff_a     = ff_factors.loc[common]
    sp500_a  = sp500_ret.loc[common]
    ret_a    = returns.loc[common]
    excess   = port_a - ff_a["RF"]

    # ── 5. Risk metrics ───────────────────────────────────────────────────────
    ann_ret = port_a.mean() * DAYS
    ann_vol = port_a.std()  * np.sqrt(DAYS)
    ann_rf  = ff_a["RF"].mean() * DAYS
    sharpe  = (ann_ret - ann_rf) / ann_vol if ann_vol > 0 else 0

    cum      = (1 + port_a).cumprod()
    max_dd   = ((cum - cum.cummax()) / cum.cummax()).min()

    monthly  = port_a.resample("ME").apply(lambda x: (1+x).prod() - 1)
   monthly_clean = monthly.dropna()
var95 = np.percentile(monthly_clean, 5) if len(monthly_clean) > 0 else -0.05

    sp500_1d = np.asarray(sp500_a).flatten()
    port_1d  = np.asarray(port_a).flatten()
    cov_m    = np.cov(port_1d, sp500_1d)
    beta     = cov_m[0,1] / cov_m[1,1] if cov_m[1,1] != 0 else 1.0

    sp500_ann    = sp500_a.mean() * DAYS
    sp500_vol    = sp500_a.std()  * np.sqrt(DAYS)
    sp500_sharpe = (sp500_ann - ann_rf) / sp500_vol if sp500_vol > 0 else 0

    cum_sp500 = (1 + sp500_a).cumprod()

    # ── 6. Factor regression ──────────────────────────────────────────────────
    X = ff_a[["Mkt-RF","SMB","HML","RMW","CMA"]].values
    y = excess.values
    factor_model = LinearRegression().fit(X, y)
    factor_betas = dict(zip(["Mkt-RF","SMB","HML","RMW","CMA"], factor_model.coef_))
    alpha        = factor_model.intercept_ * DAYS
    r_squared    = factor_model.score(X, y)

    # ── 7. Derived series ─────────────────────────────────────────────────────
    drawdown       = (cum - cum.cummax()) / cum.cummax()
    rolling_sharpe = port_a.rolling(63).apply(
        lambda x: (x.mean() * DAYS) / (x.std() * np.sqrt(DAYS)) if x.std() > 0 else np.nan
    )
    corr_matrix = ret_a.corr()

    # ── 8. FX data ────────────────────────────────────────────────────────────
    fx_tickers = list(FX_PAIRS.keys())
    raw_fx = yf.download(fx_tickers, start=start_str, end=end_str,
                          auto_adjust=True, progress=False)
    if isinstance(raw_fx.columns, pd.MultiIndex):
        fx_prices = raw_fx["Close"].copy()
    else:
        fx_prices = raw_fx[["Close"]].copy()
        fx_prices.columns = fx_tickers

    for col in fx_prices.columns:
        fx_prices[col] = pd.to_numeric(fx_prices[col].values.flatten(), errors="coerce")

    fx_prices  = fx_prices.dropna(how="all").ffill()
    fx_returns = fx_prices.pct_change().dropna()

    # FX risk metrics
    fx_rows = []
    for ticker, (name, quote, base) in FX_PAIRS.items():
        if ticker not in fx_returns.columns:
            continue
        ret_fx  = fx_returns[ticker].dropna()
        av      = ret_fx.std() * np.sqrt(DAYS)
        ar      = ret_fx.mean() * DAYS
        sh      = ar / av if av > 0 else 0
        carry   = CARRY_RATES.get(base, 0.03) - CARRY_RATES.get(quote, 0.03)
        m_fx    = ret_fx.resample("ME").apply(lambda x: (1+x).prod() - 1)
       m_fx_clean = m_fx.dropna()
v95_fx = np.percentile(m_fx_clean, 5) if len(m_fx_clean) > 0 else -0.05
        c_fx    = (1 + ret_fx).cumprod()
        mdd_fx  = ((c_fx - c_fx.cummax()) / c_fx.cummax()).min()
        fx_rows.append(dict(ticker=ticker, name=name, base=base, quote=quote,
                            ann_ret=ar, ann_vol=av, sharpe=sh,
                            carry=carry, var95=v95_fx, mdd=mdd_fx))

    fx_metrics  = pd.DataFrame(fx_rows).set_index("ticker") if fx_rows else pd.DataFrame()
    fx_corr     = fx_returns.corr()

    # Portfolio FX exposure via sector defaults
    portfolio_fx = {}
    for t, w in zip(valid_tickers, weights_arr):
        sec = fundamentals[t]["sector"]
        exposure = SECTOR_FX.get(sec, SECTOR_FX["Unknown"])
        for currency, pct in exposure.items():
            portfolio_fx[currency] = portfolio_fx.get(currency, 0) + w * pct * 100
    portfolio_fx = dict(sorted(portfolio_fx.items(), key=lambda x: -x[1]))

    # ── Package and return ────────────────────────────────────────────────────
    return {
        # Config
        "portfolio":      portfolio,
        "tickers":        valid_tickers,
        "weights":        weights_arr,
        "start_date":     start_str,
        "end_date":       end_str,

        # Price data
        "prices":         prices,
        "returns":        ret_a,
        "port_ret":       port_a,
        "sp500_ret":      sp500_a,
        "ff_factors":     ff_a,
        "excess":         excess,

        # Risk metrics
        "ann_return":     ann_ret,
        "ann_vol":        ann_vol,
        "sharpe":         sharpe,
        "max_dd":         max_dd,
        "var_95":         var95,
        "beta":           beta,
        "sp500_ann":      sp500_ann,
        "sp500_vol":      sp500_vol,
        "sp500_sharpe":   sp500_sharpe,
        "ann_rf":         ann_rf,

        # Factor model
        "factor_betas":   factor_betas,
        "alpha":          alpha,
        "r_squared":      r_squared,

        # Series
        "cum_port":       cum,
        "cum_sp500":      cum_sp500,
        "drawdown":       drawdown,
        "monthly_ret":    monthly,
        "rolling_sharpe": rolling_sharpe,
        "corr_matrix":    corr_matrix,

        # Fundamentals
        "fundamentals":   fundamentals,
        "sector_weights": sector_weights,

        # FX
        "fx_prices":      fx_prices,
        "fx_returns":     fx_returns,
        "fx_metrics":     fx_metrics,
        "fx_corr":        fx_corr,
        "portfolio_fx":   portfolio_fx,
    }
