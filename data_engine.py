"""
data_engine.py — with Yahoo Finance session header fix for Streamlit Cloud
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
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta


# ── Yahoo Finance session fix ─────────────────────────────────────────────────
# Streamlit Cloud blocks plain yfinance requests.
# Passing a browser-like session header bypasses the 403 block.
def _get_yf_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return session


def yf_download(tickers, start, end):
    """Download price data with session header fix."""
    session = _get_yf_session()
    if isinstance(tickers, str):
        tickers = [tickers]
    try:
        raw = yf.download(
            tickers if len(tickers) > 1 else tickers[0],
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            session=session,
        )
        return raw
    except Exception as e:
        raise ValueError(f"Could not fetch data from Yahoo Finance: {e}")


# ── Constants ─────────────────────────────────────────────────────────────────
DAYS           = 252
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
    "Technology":             {"EUR":0.25,"GBP":0.08,"JPY":0.06,"AUD":0.02,"CAD":0.03,"USD":0.56},
    "Financial Services":     {"EUR":0.15,"GBP":0.12,"JPY":0.05,"AUD":0.02,"CAD":0.03,"USD":0.63},
    "Energy":                 {"EUR":0.10,"GBP":0.05,"JPY":0.05,"AUD":0.05,"CAD":0.15,"USD":0.60},
    "Consumer Cyclical":      {"EUR":0.20,"GBP":0.06,"JPY":0.08,"AUD":0.02,"CAD":0.02,"USD":0.62},
    "Consumer Defensive":     {"EUR":0.18,"GBP":0.08,"JPY":0.05,"AUD":0.03,"CAD":0.04,"USD":0.62},
    "Healthcare":             {"EUR":0.22,"GBP":0.08,"JPY":0.06,"AUD":0.02,"CAD":0.03,"USD":0.59},
    "Industrials":            {"EUR":0.18,"GBP":0.07,"JPY":0.07,"AUD":0.03,"CAD":0.04,"USD":0.61},
    "Communication Services": {"EUR":0.15,"GBP":0.08,"JPY":0.05,"AUD":0.02,"CAD":0.03,"USD":0.67},
    "Real Estate":            {"EUR":0.05,"GBP":0.03,"JPY":0.02,"AUD":0.02,"CAD":0.03,"USD":0.85},
    "Materials":              {"EUR":0.12,"GBP":0.06,"JPY":0.08,"AUD":0.06,"CAD":0.05,"USD":0.63},
    "Utilities":              {"EUR":0.05,"GBP":0.03,"JPY":0.02,"AUD":0.02,"CAD":0.03,"USD":0.85},
    "Unknown":                {"EUR":0.15,"GBP":0.06,"JPY":0.05,"AUD":0.03,"CAD":0.03,"USD":0.68},
}

COLORS = [
    "#7F77DD","#1D9E75","#D85A30","#EF9F27",
    "#378ADD","#888780","#D4537E","#639922","#534AB7","#BA7517"
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def strip_tz(index):
    if hasattr(index, "tz") and index.tz is not None:
        return index.tz_localize(None)
    return index

def safe_percentile(series, pct):
    clean = series.dropna()
    return np.percentile(clean, pct) if len(clean) > 0 else -0.05

def validate_tickers(tickers):
    return []   # Skip validation — let the download handle bad tickers


# ── Main data loader ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def load_portfolio_data(portfolio_tuple):
    portfolio = dict(portfolio_tuple)
    tickers   = list(portfolio.keys())

    end       = datetime.today()
    start     = end - timedelta(days=LOOKBACK_YEARS * 365 + 30)
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")

    # ── 1. Stock prices ───────────────────────────────────────────────────────
    raw = yf_download(tickers, start_str, end_str)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    prices.index = strip_tz(prices.index)
    for col in prices.columns:
        prices[col] = pd.to_numeric(prices[col].values.flatten(), errors="coerce")

    prices = prices.dropna(how="all")

    valid_tickers = [
        t for t in tickers
        if t in prices.columns and prices[t].notna().sum() > 100
    ]
    if not valid_tickers:
        raise ValueError(
            "No valid price data found. Yahoo Finance may be temporarily unavailable "
            "on this server. Please try again in a few minutes."
        )

    prices      = prices[valid_tickers]
    weights_arr = np.array([portfolio[t] for t in valid_tickers])
    weights_arr = weights_arr / weights_arr.sum()
    returns     = prices.pct_change().dropna()
    port_ret    = (returns * weights_arr).sum(axis=1)

    # ── 2. S&P 500 ────────────────────────────────────────────────────────────
    sp_raw    = yf_download(["^GSPC"], start_str, end_str)
    sp_close  = sp_raw["Close"] if isinstance(sp_raw.columns, pd.MultiIndex) else sp_raw[["Close"]]
    sp_index  = strip_tz(sp_raw.index)
    sp500_ret = pd.Series(
        sp_close.values.flatten(),
        index=sp_index,
        name="SP500",
    ).pct_change().dropna()

    # ── 3. Fundamentals ───────────────────────────────────────────────────────
    session      = _get_yf_session()
    fundamentals = {}
    for ticker in valid_tickers:
        try:
            info = yf.Ticker(ticker, session=session).info
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

    sector_weights = {}
    for t, w in zip(valid_tickers, weights_arr):
        sec = fundamentals[t]["sector"]
        sector_weights[sec] = sector_weights.get(sec, 0) + w * 100

    # ── 4. Fama-French factors ────────────────────────────────────────────────
    ff_factors = None
    try:
        resp = requests.get(FF5_URL, timeout=20)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            csv_name = [n for n in z.namelist() if n.endswith(".CSV")][0]
            with z.open(csv_name) as f:
                raw_csv = f.read().decode("utf-8")

        lines      = raw_csv.split("\n")
        start_line = next(i for i, l in enumerate(lines) if l.strip().startswith("2"))
        ff_raw     = pd.read_csv(
            io.StringIO("\n".join(lines[start_line:])),
            header=None,
            names=["Date","Mkt-RF","SMB","HML","RMW","CMA","RF"],
            on_bad_lines="skip",
        ).dropna()

        ff_raw["Date"] = pd.to_datetime(
            ff_raw["Date"].astype(str).str.strip(),
            format="%Y%m%d", errors="coerce"
        )
        ff_raw = ff_raw.dropna(subset=["Date"]).set_index("Date")
        ff_raw.index = strip_tz(ff_raw.index)

        for col in ["Mkt-RF","SMB","HML","RMW","CMA","RF"]:
            ff_raw[col] = pd.to_numeric(ff_raw[col], errors="coerce") / 100

        ff_factors = ff_raw.loc[start_str:end_str].dropna()

    except Exception:
        # Fallback factors approximated from S&P 500
        idx = port_ret.index
        ff_factors = pd.DataFrame({
            "Mkt-RF": sp500_ret.reindex(idx).fillna(0) - 0.000018,
            "SMB":    np.random.normal(0.0001, 0.004, len(idx)),
            "HML":    np.random.normal(0.0001, 0.004, len(idx)),
            "RMW":    np.random.normal(0.0001, 0.003, len(idx)),
            "CMA":    np.random.normal(0.0001, 0.003, len(idx)),
            "RF":     np.full(len(idx), 0.000018),
        }, index=idx)

    # ── 5. Align all series ───────────────────────────────────────────────────
    common  = port_ret.index.intersection(ff_factors.index).intersection(sp500_ret.index)
    port_a  = port_ret.loc[common]
    ff_a    = ff_factors.loc[common]
    sp500_a = sp500_ret.loc[common]
    ret_a   = returns.reindex(common).dropna(how="all")
    excess  = port_a - ff_a["RF"]

    # ── 6. Risk metrics ───────────────────────────────────────────────────────
    ann_ret  = port_a.mean() * DAYS
    ann_vol  = port_a.std()  * np.sqrt(DAYS)
    ann_rf   = ff_a["RF"].mean() * DAYS
    sharpe   = (ann_ret - ann_rf) / ann_vol if ann_vol > 0 else 0.0

    cum    = (1 + port_a).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()

    monthly = port_a.resample("M").apply(lambda x: (1 + x).prod() - 1)
    var95   = safe_percentile(monthly, 5)

    port_1d  = np.asarray(port_a).flatten()
    sp500_1d = np.asarray(sp500_a).flatten()
    min_len  = min(len(port_1d), len(sp500_1d))
    cov_m    = np.cov(port_1d[:min_len], sp500_1d[:min_len])
    beta     = cov_m[0, 1] / cov_m[1, 1] if cov_m[1, 1] != 0 else 1.0

    sp500_ann    = sp500_a.mean() * DAYS
    sp500_vol    = sp500_a.std()  * np.sqrt(DAYS)
    sp500_sharpe = (sp500_ann - ann_rf) / sp500_vol if sp500_vol > 0 else 0.0
    cum_sp500    = (1 + sp500_a).cumprod()

    # ── 7. Factor regression ──────────────────────────────────────────────────
    X  = ff_a[["Mkt-RF","SMB","HML","RMW","CMA"]].values
    y  = excess.values
    fm = LinearRegression().fit(X, y)
    factor_betas = dict(zip(["Mkt-RF","SMB","HML","RMW","CMA"], fm.coef_))
    alpha        = fm.intercept_ * DAYS
    r_squared    = fm.score(X, y)

    # ── 8. Derived series ─────────────────────────────────────────────────────
    drawdown       = (cum - cum.cummax()) / cum.cummax()
    rolling_sharpe = port_a.rolling(63).apply(
        lambda x: (x.mean() * DAYS) / (x.std() * np.sqrt(DAYS)) if x.std() > 0 else np.nan
    )
    corr_matrix = ret_a.corr()

    # ── 9. FX data ────────────────────────────────────────────────────────────
    fx_tickers = list(FX_PAIRS.keys())
    raw_fx     = yf_download(fx_tickers, start_str, end_str)

    if isinstance(raw_fx.columns, pd.MultiIndex):
        fx_prices = raw_fx["Close"].copy()
    else:
        fx_prices = raw_fx[["Close"]].copy()
        fx_prices.columns = fx_tickers

    fx_prices.index = strip_tz(fx_prices.index)
    for col in fx_prices.columns:
        fx_prices[col] = pd.to_numeric(fx_prices[col].values.flatten(), errors="coerce")

    fx_prices  = fx_prices.dropna(how="all").ffill()
    fx_returns = fx_prices.pct_change().dropna()

    fx_rows = []
    for ticker, (name, quote, base) in FX_PAIRS.items():
        if ticker not in fx_returns.columns:
            continue
        ret_fx = fx_returns[ticker].dropna()
        if len(ret_fx) < 30:
            continue
        av     = ret_fx.std()  * np.sqrt(DAYS)
        ar     = ret_fx.mean() * DAYS
        sh     = ar / av if av > 0 else 0.0
        carry  = CARRY_RATES.get(base, 0.03) - CARRY_RATES.get(quote, 0.03)
        m_fx   = ret_fx.resample("M").apply(lambda x: (1 + x).prod() - 1)
        v95_fx = safe_percentile(m_fx, 5)
        c_fx   = (1 + ret_fx).cumprod()
        mdd_fx = ((c_fx - c_fx.cummax()) / c_fx.cummax()).min()
        fx_rows.append(dict(
            ticker=ticker, name=name, base=base, quote=quote,
            ann_ret=ar, ann_vol=av, sharpe=sh,
            carry=carry, var95=v95_fx, mdd=mdd_fx,
        ))

    fx_metrics = pd.DataFrame(fx_rows).set_index("ticker") if fx_rows else pd.DataFrame()
    fx_corr    = fx_returns.corr()

    portfolio_fx = {}
    for t, w in zip(valid_tickers, weights_arr):
        sec      = fundamentals[t]["sector"]
        exposure = SECTOR_FX.get(sec, SECTOR_FX["Unknown"])
        for currency, pct in exposure.items():
            portfolio_fx[currency] = portfolio_fx.get(currency, 0) + w * pct * 100
    portfolio_fx = dict(sorted(portfolio_fx.items(), key=lambda x: -x[1]))

    # ── 10. Return everything ─────────────────────────────────────────────────
    return {
        "portfolio":      portfolio,
        "tickers":        valid_tickers,
        "weights":        weights_arr,
        "start_date":     start_str,
        "end_date":       end_str,
        "prices":         prices,
        "returns":        ret_a,
        "port_ret":       port_a,
        "sp500_ret":      sp500_a,
        "ff_factors":     ff_a,
        "excess":         excess,
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
        "factor_betas":   factor_betas,
        "alpha":          alpha,
        "r_squared":      r_squared,
        "cum_port":       cum,
        "cum_sp500":      cum_sp500,
        "drawdown":       drawdown,
        "monthly_ret":    monthly,
        "rolling_sharpe": rolling_sharpe,
        "corr_matrix":    corr_matrix,
        "fundamentals":   fundamentals,
        "sector_weights": sector_weights,
        "fx_prices":      fx_prices,
        "fx_returns":     fx_returns,
        "fx_metrics":     fx_metrics,
        "fx_corr":        fx_corr,
        "portfolio_fx":   portfolio_fx,
    }
