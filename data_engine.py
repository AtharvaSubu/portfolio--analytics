"""
data_engine.py
All data fetching and number-crunching lives here. Pages only draw charts.

Speed design:
  * Every network call is cached separately and shared across all visitors,
    so the second person to open the app gets FX and factor data instantly.
  * Stocks + S&P 500 are fetched in ONE download call.
  * Company fundamentals are fetched in parallel instead of one by one.
  * Global FX data does not depend on the portfolio, so the FX page works
    even before anyone enters tickers.
"""

import io
import zipfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
DAYS = 252
LOOKBACK_YEARS = 5
BENCHMARK = "^GSPC"
DOLLAR_INDEX = "DX-Y.NYB"

FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)

COLORS = [
    "#7F77DD", "#1D9E75", "#D85A30", "#EF9F27", "#378ADD",
    "#888780", "#D4537E", "#639922", "#534AB7", "#BA7517",
]

# Global currency universe.
# invert=True means Yahoo quotes it as USD/XXX, so we flip it so that
# a positive number always means "this currency got stronger vs the dollar".
CURRENCIES = {
    "EUR": {"name": "Euro",              "ticker": "EURUSD=X", "invert": False, "region": "Developed"},
    "GBP": {"name": "British Pound",     "ticker": "GBPUSD=X", "invert": False, "region": "Developed"},
    "JPY": {"name": "Japanese Yen",      "ticker": "USDJPY=X", "invert": True,  "region": "Developed"},
    "CHF": {"name": "Swiss Franc",       "ticker": "USDCHF=X", "invert": True,  "region": "Developed"},
    "CAD": {"name": "Canadian Dollar",   "ticker": "USDCAD=X", "invert": True,  "region": "Developed"},
    "AUD": {"name": "Australian Dollar", "ticker": "AUDUSD=X", "invert": False, "region": "Developed"},
    "NZD": {"name": "New Zealand Dollar","ticker": "NZDUSD=X", "invert": False, "region": "Developed"},
    "CNY": {"name": "Chinese Yuan",      "ticker": "USDCNY=X", "invert": True,  "region": "Emerging"},
    "INR": {"name": "Indian Rupee",      "ticker": "USDINR=X", "invert": True,  "region": "Emerging"},
    "KRW": {"name": "South Korean Won",  "ticker": "USDKRW=X", "invert": True,  "region": "Emerging"},
    "MXN": {"name": "Mexican Peso",      "ticker": "USDMXN=X", "invert": True,  "region": "Emerging"},
    "BRL": {"name": "Brazilian Real",    "ticker": "USDBRL=X", "invert": True,  "region": "Emerging"},
    "ZAR": {"name": "South African Rand","ticker": "USDZAR=X", "invert": True,  "region": "Emerging"},
    "TRY": {"name": "Turkish Lira",      "ticker": "USDTRY=X", "invert": True,  "region": "Emerging"},
}

# Central bank policy rates — defaults as of early Nov 2026 (global-rates.com).
# Users can edit these in the app; update here when banks move.
POLICY_RATES = {
    "USD": 4.00, "EUR": 2.65, "GBP": 4.00, "JPY": 1.25, "CHF": 0.00,
    "CAD": 2.50, "AUD": 4.85, "NZD": 2.75, "CNY": 3.00, "INR": 5.50,
    "KRW": 3.00, "MXN": 6.25, "BRL": 13.75, "ZAR": 7.25, "TRY": 36.00,
}
POLICY_RATES_AS_OF = "early November 2026"

# Typical revenue mix by currency for each sector (rough estimate, not company data)
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
    "Basic Materials":        {"EUR":0.12,"GBP":0.06,"JPY":0.08,"AUD":0.06,"CAD":0.05,"USD":0.63},
    "Utilities":              {"EUR":0.05,"GBP":0.03,"JPY":0.02,"AUD":0.02,"CAD":0.03,"USD":0.85},
    "Unknown":                {"EUR":0.15,"GBP":0.06,"JPY":0.05,"AUD":0.03,"CAD":0.03,"USD":0.68},
}


# ── Small helpers ─────────────────────────────────────────────────────────────
def date_window(years=LOOKBACK_YEARS):
    end = datetime.today() + timedelta(days=1)
    start = end - timedelta(days=years * 365 + 30)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def strip_tz(index):
    if getattr(index, "tz", None) is not None:
        return index.tz_localize(None)
    return index


def month_end(series):
    """Monthly compounded returns. Works on old ('M') and new ('ME') pandas."""
    for freq in ("ME", "M"):
        try:
            return series.resample(freq).apply(lambda x: (1 + x).prod() - 1)
        except (ValueError, KeyError):
            continue
    raise ValueError("Could not resample to month end")


def week_end(series_or_df):
    return (1 + series_or_df).resample("W-FRI").prod() - 1


def safe_percentile(series, pct, fallback=-0.05):
    clean = pd.Series(series).dropna()
    return float(np.percentile(clean, pct)) if len(clean) else fallback


def _download_close(tickers, start, end):
    """Download closing prices for a list of tickers as a clean wide DataFrame."""
    tickers = list(dict.fromkeys(tickers))  # de-duplicate, keep order
    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, threads=True, group_by="column",
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = tickers[:1]

    close.index = strip_tz(pd.to_datetime(close.index))
    close = close.apply(pd.to_numeric, errors="coerce")
    close = close.loc[:, close.notna().sum() > 0]
    return close.sort_index()


# ── Cached network calls (shared across all visitors) ─────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices(tickers, start, end):
    return _download_close(list(tickers) + [BENCHMARK], start, end)


def _one_fundamental(ticker):
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    return ticker, {
        "company":    info.get("longName") or info.get("shortName") or ticker,
        "sector":     info.get("sector") or "Unknown",
        "market_cap": info.get("marketCap") or 0,
        "pe_ratio":   info.get("trailingPE") or 0,
    }


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_fundamentals(tickers):
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(tickers)))) as pool:
        return dict(pool.map(_one_fundamental, tickers))


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_ff_factors():
    """Fama-French 5 factors (daily, decimals). Returns None if unreachable."""
    try:
        resp = requests.get(FF5_URL, timeout=20)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            text = z.read(name).decode("utf-8", errors="ignore")
        lines = text.splitlines()
        first = next(i for i, l in enumerate(lines) if l.strip()[:1].isdigit())
        df = pd.read_csv(
            io.StringIO("\n".join(lines[first:])), header=None,
            names=["Date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
            on_bad_lines="skip",
        )
        df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(),
                                    format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
        df = df.apply(pd.to_numeric, errors="coerce").dropna() / 100
        return df
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_global_fx(start, end):
    """
    Returns (strength, dxy):
      strength — each currency's value in USD (positive move = currency stronger)
      dxy      — US Dollar Index level (may be empty if Yahoo has no data)
    """
    tickers = [c["ticker"] for c in CURRENCIES.values()] + [DOLLAR_INDEX]
    close = _download_close(tickers, start, end)
    if close.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    strength = {}
    for code, meta in CURRENCIES.items():
        if meta["ticker"] in close.columns:
            s = close[meta["ticker"]].replace(0, np.nan)
            strength[code] = 1 / s if meta["invert"] else s
    strength = pd.DataFrame(strength).ffill()

    if DOLLAR_INDEX in close.columns and close[DOLLAR_INDEX].notna().sum() > 100:
        dxy = close[DOLLAR_INDEX].ffill()
    else:
        # Fallback: equal-weight dollar basket (dollar up = currencies down)
        majors = [c for c in ["EUR", "JPY", "GBP", "CAD", "CHF", "AUD"] if c in strength]
        rets = -strength[majors].pct_change().mean(axis=1).fillna(0)
        dxy = 100 * (1 + rets).cumprod()
    return strength, dxy.rename("USD Index")


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_custom_pair(ticker, start, end):
    close = _download_close([ticker], start, end)
    if close.empty or close.iloc[:, 0].notna().sum() < 60:
        return None
    return close.iloc[:, 0].rename(ticker)


# ── Global FX analytics (no portfolio needed) ─────────────────────────────────
def period_changes(strength):
    """% change of each currency vs USD over standard look-back windows."""
    last = strength.ffill().iloc[-1]
    today = strength.index[-1]
    windows = {
        "1W": today - pd.Timedelta(days=7),
        "1M": today - pd.DateOffset(months=1),
        "3M": today - pd.DateOffset(months=3),
        "YTD": pd.Timestamp(year=today.year, month=1, day=1) - pd.Timedelta(days=1),
        "1Y": today - pd.DateOffset(years=1),
    }
    out = {}
    for label, when in windows.items():
        past = strength.loc[:when].ffill()
        out[label] = (last / past.iloc[-1] - 1) if len(past) else np.nan
    return pd.DataFrame(out)


def risk_table(strength, rates):
    """Volatility, stress, drawdown and carry for every currency."""
    rets = strength.pct_change().dropna(how="all")
    one_y = rets.loc[rets.index[-1] - pd.DateOffset(years=1):]
    rows = []
    usd_rate = rates.get("USD", 0)
    for code in strength.columns:
        r = rets[code].dropna()
        if len(r) < 60:
            continue
        vol_1m = r.tail(21).std() * np.sqrt(DAYS)
        vol_1y = one_y[code].std() * np.sqrt(DAYS)
        level = strength[code].dropna()
        peak_1y = level.loc[level.index[-1] - pd.DateOffset(years=1):].max()
        carry = (rates.get(code, np.nan) - usd_rate) / 100
        rows.append({
            "code": code,
            "name": CURRENCIES[code]["name"],
            "region": CURRENCIES[code]["region"],
            "vol_1m": vol_1m,
            "vol_1y": vol_1y,
            "stress": vol_1m / vol_1y if vol_1y > 0 else np.nan,
            "off_peak": level.iloc[-1] / peak_1y - 1,
            "var95_month": safe_percentile(month_end(r), 5),
            "carry": carry,
            "carry_to_risk": carry / vol_1y if vol_1y > 0 else np.nan,
        })
    return pd.DataFrame(rows).set_index("code")


def regime(vol):
    if vol < 0.06:
        return "Calm"
    if vol < 0.12:
        return "Elevated"
    return "Stormy"


# ── Portfolio analytics ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def load_portfolio_data(portfolio_tuple):
    portfolio = dict(portfolio_tuple)
    tickers = tuple(portfolio.keys())
    start, end = date_window()

    # 1. Prices (stocks + benchmark in one call)
    prices_all = fetch_prices(tickers, start, end)
    if prices_all.empty or BENCHMARK not in prices_all.columns:
        raise ValueError(
            "Yahoo Finance did not return data right now. Please wait a minute and try again."
        )

    valid = [t for t in tickers if t in prices_all.columns and prices_all[t].notna().sum() > 120]
    dropped = [t for t in tickers if t not in valid]
    if len(valid) == 0:
        raise ValueError(
            f"No price history found for: {', '.join(tickers)}. "
            "Check the symbols (e.g. AAPL, RELIANCE.NS, BP.L)."
        )

    prices = prices_all[valid].ffill()
    weights = np.array([portfolio[t] for t in valid], dtype=float)
    weights = weights / weights.sum()

    returns = prices.pct_change().dropna(how="all").fillna(0)
    port_ret = (returns * weights).sum(axis=1)
    sp500_ret = prices_all[BENCHMARK].ffill().pct_change().dropna()

    # 2. Fundamentals (parallel, cached a day)
    fundamentals = fetch_fundamentals(tuple(valid))
    sector_weights = {}
    for t, w in zip(valid, weights):
        sec = fundamentals[t]["sector"]
        sector_weights[sec] = sector_weights.get(sec, 0) + w * 100

    # 3. Factors — real Fama-French if available, else market-only model
    ff = fetch_ff_factors()
    ff_source = "Fama-French 5-factor (Ken French Data Library)"
    if ff is None or ff.loc[start:].empty:
        rf_daily = 0.04 / DAYS
        ff = pd.DataFrame({"Mkt-RF": sp500_ret - rf_daily, "RF": rf_daily}, index=sp500_ret.index)
        ff_source = "Market-only model (Fama-French data unavailable right now)"
    factor_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"] if c in ff.columns]

    # 4. Align
    common = port_ret.index.intersection(sp500_ret.index).intersection(ff.index)
    if len(common) < 120:
        # Fama-French lags a month or two; fall back to aligning without it
        common = port_ret.index.intersection(sp500_ret.index)
        rf_daily = 0.04 / DAYS
        ff = pd.DataFrame({"Mkt-RF": sp500_ret - rf_daily, "RF": rf_daily}, index=sp500_ret.index)
        factor_cols = ["Mkt-RF"]
        ff_source = "Market-only model (Fama-French data not yet available for this period)"

    port_a, sp_a = port_ret.loc[common], sp500_ret.loc[common]
    ff_a, ret_a = ff.loc[common], returns.loc[common]
    excess = port_a - ff_a["RF"]

    # 5. Risk metrics
    ann_rf = ff_a["RF"].mean() * DAYS
    ann_ret = port_a.mean() * DAYS
    ann_vol = port_a.std() * np.sqrt(DAYS)
    sharpe = (ann_ret - ann_rf) / ann_vol if ann_vol > 0 else 0.0

    cum = (1 + port_a).cumprod()
    drawdown = cum / cum.cummax() - 1
    monthly = month_end(port_a)
    var95 = safe_percentile(monthly, 5)

    cov = np.cov(port_a.values, sp_a.values)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] else 1.0

    sp_ann = sp_a.mean() * DAYS
    sp_vol = sp_a.std() * np.sqrt(DAYS)
    sp_sharpe = (sp_ann - ann_rf) / sp_vol if sp_vol > 0 else 0.0

    roll_mean = port_a.rolling(63).mean()
    roll_std = port_a.rolling(63).std()
    rolling_sharpe = (roll_mean * DAYS) / (roll_std * np.sqrt(DAYS))

    # 6. Factor regressions — portfolio and each stock (computed once, cached)
    X = ff_a[factor_cols].values
    fm = LinearRegression().fit(X, excess.values)
    factor_betas = dict(zip(factor_cols, fm.coef_))

    stock_rows = []
    for t in valid:
        y = (ret_a[t] - ff_a["RF"]).values
        m = LinearRegression().fit(X, y)
        row = {"Ticker": t, **dict(zip(factor_cols, m.coef_))}
        row["Alpha (ann.)"] = m.intercept_ * DAYS
        row["R²"] = m.score(X, y)
        stock_rows.append(row)
    stock_factors = pd.DataFrame(stock_rows).set_index("Ticker")

    roll_r2 = []
    window = 126
    for i in range(window, len(excess), 5):  # every 5 days is plenty for a chart
        Xi, yi = X[i - window:i], excess.values[i - window:i]
        roll_r2.append((excess.index[i], LinearRegression().fit(Xi, yi).score(Xi, yi)))
    rolling_r2 = pd.Series(dict(roll_r2))

    return {
        "portfolio": portfolio, "tickers": valid, "dropped": dropped, "weights": weights,
        "start_date": common[0].strftime("%Y-%m-%d"), "end_date": common[-1].strftime("%Y-%m-%d"),
        "returns": ret_a, "port_ret": port_a, "sp500_ret": sp_a, "ff_factors": ff_a, "excess": excess,
        "ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": drawdown.min(),
        "var_95": var95, "beta": beta, "ann_rf": ann_rf,
        "sp500_ann": sp_ann, "sp500_vol": sp_vol, "sp500_sharpe": sp_sharpe,
        "factor_betas": factor_betas, "alpha": fm.intercept_ * DAYS,
        "r_squared": fm.score(X, excess.values), "ff_source": ff_source,
        "stock_factors": stock_factors, "rolling_r2": rolling_r2,
        "cum_port": cum, "cum_sp500": (1 + sp_a).cumprod(), "drawdown": drawdown,
        "monthly_ret": monthly, "rolling_sharpe": rolling_sharpe,
        "corr_matrix": ret_a.corr(),
        "fundamentals": fundamentals, "sector_weights": sector_weights,
    }


def estimated_fx_exposure(tickers, weights, fundamentals):
    """Sector-based estimate of revenue by currency, in % of portfolio."""
    out = {}
    for t, w in zip(tickers, weights):
        mix = SECTOR_FX.get(fundamentals[t]["sector"], SECTOR_FX["Unknown"])
        for cur, share in mix.items():
            out[cur] = out.get(cur, 0) + w * share * 100
    return dict(sorted(out.items(), key=lambda x: -x[1]))


def dollar_sensitivity(port_ret, sp500_ret, stock_returns, dxy):
    """
    Measured (not assumed) dollar sensitivity, from weekly returns:
        portfolio = a + b_mkt * S&P 500 + b_usd * Dollar Index
    b_usd = how much the portfolio moves per 1% dollar move, beyond the market.
    """
    dxy_ret = dxy.pct_change()
    df = pd.concat([port_ret.rename("port"), sp500_ret.rename("mkt"),
                    dxy_ret.rename("usd")], axis=1).dropna()
    wk = week_end(df).dropna()
    if len(wk) < 52:
        return None

    X = wk[["mkt", "usd"]].values
    m = LinearRegression().fit(X, wk["port"].values)

    per_stock = {}
    stocks_wk = week_end(stock_returns.loc[df.index]).reindex(wk.index)
    for t in stocks_wk.columns:
        y = stocks_wk[t].fillna(0).values
        per_stock[t] = LinearRegression().fit(X, y).coef_[1]

    rolling_corr = wk["port"].rolling(52).corr(wk["usd"]).dropna()

    return {
        "usd_beta": m.coef_[1],
        "mkt_beta": m.coef_[0],
        "r2": m.score(X, wk["port"].values),
        "per_stock": pd.Series(per_stock).sort_values(),
        "rolling_corr": rolling_corr,
        "weeks": len(wk),
    }
