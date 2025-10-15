
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except Exception:
    yf = None

st.set_page_config(page_title="Stock Dashboard", page_icon="📈", layout="wide")

@st.cache_data(show_spinner=False, ttl=60*15)
def load_prices(tickers, start, end):
    if yf is None:
        raise RuntimeError("yfinance is not installed. Install requirements and try again.")
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="ticker")
    return df

def ensure_close(df, tickers):
    if isinstance(df.columns, pd.MultiIndex):
        close = df.xs("Close", axis=1, level=1)
        close = close[[t for t in tickers if t in close.columns]]
    else:
        close = df[["Close"]].rename(columns={"Close": tickers[0]})
    return close

def moving_average(series, window):
    return series.rolling(window).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    signal_line = line.ewm(span=signal, adjust=False).mean()
    hist = line - signal_line
    return line, signal_line, hist

def max_drawdown(series):
    roll_max = series.cummax()
    drawdown = series/roll_max - 1.0
    return drawdown.min()

def sharpe_ratio(returns, risk_free=0.0, periods_per_year=252):
    ex = returns - risk_free/periods_per_year
    if ex.std() == 0:
        return np.nan
    return np.sqrt(periods_per_year) * ex.mean() / ex.std()

with st.sidebar:
    st.header("⚙️ Controls")
    default_tickers = ["AAPL", "MSFT", "GOOGL"]
    tickers = st.text_input("Tickers (comma-separated)", value=",".join(default_tickers))
    tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    today = date.today()
    start = st.date_input("Start date", today - timedelta(days=365))
    end = st.date_input("End date", today)
    uploaded = st.file_uploader("Optional: Upload CSV with Date + Price column", type=["csv"])
    st.markdown("---")
    show_ma = st.checkbox("Show Moving Averages (20/50/200)", value=True)
    show_rsi = st.checkbox("Show RSI", value=True)
    show_macd = st.checkbox("Show MACD", value=False)

st.title("📈 Stock Analysis Dashboard")
st.caption("Formal, concise analytics for quick decisions.")

data_source = "yfinance"
if uploaded is not None:
    import pandas as pd, numpy as np
    try:
        user_df = pd.read_csv(uploaded)
        price_col = None
        for c in ["Close", "Adj Close", "Price", "price", "close"]:
            if c in user_df.columns:
                price_col = c
                break
        if price_col is None:
            num_cols = user_df.select_dtypes(include=[np.number]).columns
            if len(num_cols):
                price_col = num_cols[-1]
        user_df["Date"] = pd.to_datetime(user_df["Date"])
        user_df = user_df.set_index("Date").sort_index()
        close = user_df[[price_col]].rename(columns={price_col: "UPLOADED"})
        tickers = ["UPLOADED"]
        data_source = "uploaded CSV"
    except Exception as e:
        st.warning(f"CSV parse failed: {e}. Falling back to yfinance.")
        uploaded = None

if uploaded is None:
    try:
        raw = load_prices(tickers, start, end + timedelta(days=1))
        close = ensure_close(raw, tickers).dropna(how="all")
    except Exception as e:
        st.error(f"Data load failed: {e}")
        st.stop()

st.subheader("Overview")
cols = st.columns(min(len(tickers), 4))
summary_rows = []
for i, t in enumerate(tickers):
    if t not in close.columns:
        continue
    ser = close[t].dropna()
    if ser.empty:
        continue
    daily_ret = ser.pct_change().dropna()
    total_ret = (ser.iloc[-1] / ser.iloc[0]) - 1 if len(ser) > 1 else np.nan
    mdd = max_drawdown(ser)
    sharpe = sharpe_ratio(daily_ret)

    with cols[i % len(cols)]:
        st.metric(label=f"{t} | Total Return", value=f"{total_ret*100:,.2f}%")
        st.caption(f"Max Drawdown: {mdd*100:,.2f}% • Sharpe: {sharpe:,.2f}")

    summary_rows.append({
        "Ticker": t,
        "First Date": ser.index.min().date() if len(ser) else None,
        "Last Date": ser.index.max().date() if len(ser) else None,
        "Total Return %": round(total_ret*100, 2) if not np.isnan(total_ret) else None,
        "Max Drawdown %": round(mdd*100, 2) if not np.isnan(mdd) else None,
        "Sharpe": round(sharpe, 2) if not np.isnan(sharpe) else None
    })
if summary_rows:
    st.dataframe(pd.DataFrame(summary_rows))

st.subheader("Price Chart")
tab_price, tab_rsi, tab_macd = st.tabs(["Price", "RSI", "MACD"])

with tab_price:
    fig, ax = plt.subplots(figsize=(10, 4))
    for t in tickers:
        if t in close.columns:
            ax.plot(close.index, close[t], label=t)
            if show_ma:
                for w in (20, 50, 200):
                    ma = close[t].rolling(w).mean()
                    ax.plot(close.index, ma, linestyle="--", linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

with tab_rsi:
    if show_rsi:
        fig_rsi, ax_rsi = plt.subplots(figsize=(10, 2.5))
        for t in tickers:
            if t in close.columns:
                delta = close[t].diff()
                gain = (delta.where(delta > 0, 0.0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
                rs = gain / loss
                rsiv = 100 - (100 / (1 + rs))
                ax_rsi.plot(close.index, rsiv, label=f"RSI {t}")
        ax_rsi.axhline(70, linestyle="--", linewidth=1)
        ax_rsi.axhline(30, linestyle="--", linewidth=1)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI")
        ax_rsi.grid(True, alpha=0.3)
        ax_rsi.legend(loc="best")
        st.pyplot(fig_rsi, clear_figure=True)
    else:
        st.info("Enable RSI from the sidebar to view this panel.")

with tab_macd:
    if show_macd:
        fig_macd, (ax_line, ax_hist) = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
        for t in tickers:
            if t in close.columns:
                ema_fast = close[t].ewm(span=12, adjust=False).mean()
                ema_slow = close[t].ewm(span=26, adjust=False).mean()
                line = ema_fast - ema_slow
                signal_line = line.ewm(span=9, adjust=False).mean()
                hist = line - signal_line
                ax_line.plot(close.index, line, label=f"MACD {t}")
                ax_line.plot(close.index, signal_line, linestyle="--", label=f"Signal {t}")
                ax_hist.bar(close.index, hist, width=1.0)
        ax_line.set_ylabel("MACD")
        ax_line.legend(loc="best")
        ax_line.grid(True, alpha=0.3)
        ax_hist.set_ylabel("Hist")
        ax_hist.grid(True, alpha=0.3)
        st.pyplot(fig_macd, clear_figure=True)
    else:
        st.info("Enable MACD from the sidebar to view this panel.")

st.subheader("Normalized Performance")
norm = close / close.iloc[0] * 100.0
fig_norm, ax_norm = plt.subplots(figsize=(10, 3.5))
for t in tickers:
    if t in norm.columns:
        ax_norm.plot(norm.index, norm[t], label=t)
ax_norm.set_ylabel("Indexed (100 = start)")
ax_norm.legend(loc="best")
ax_norm.grid(True, alpha=0.3)
st.pyplot(fig_norm, clear_figure=True)

st.subheader("Data")
st.dataframe(close.tail(200))
csv = close.to_csv(index=True).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name="prices.csv", mime="text/csv")

st.caption(f"Source: {data_source}. Built with Streamlit + matplotlib.")
