# ========================================
# main.py  —  نسخه اصلاح شده کامل
# ========================================

import os, time, requests, math
from functools import wraps
from typing import List
import numpy as np
import pandas as pd

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from yahooquery import Ticker
import feedparser

# -------------------------------------------------------
#   PROXY ثابت برای بایننس  (Cloudflare Worker شما)
# -------------------------------------------------------
BINANCE_PROXY = "https://pro.bagheryane.workers.dev"

# API TwelveData
TWELVE_API = os.getenv("TWELVEDATA_API_KEY", "").strip()

app = FastAPI(title="Hybrid Trading Analysis - MultiSource")

# ---------- Simple CACHE ----------
_cache = {}
def cache_ttl(ttl=30):
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (func.__name__, args, tuple(sorted(kwargs.items())))
            now = time.time()
            if key in _cache:
                val, exp = _cache[key]
                if now < exp:
                    return val
            val = func(*args, **kwargs)
            _cache[key] = (val, now + ttl)
            return val
        return wrapper
    return deco


# ---------- Indicators ----------
def EMA(s, span): return s.ewm(span=span, adjust=False).mean()
def SMA(s, window): return s.rolling(window=window).mean()

def RSI(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=period - 1, adjust=True, min_periods=period).mean()
    ma_down = down.ewm(com=period - 1, adjust=True, min_periods=period).mean()
    rs = ma_up / ma_down
    rs[ma_down == 0] = np.inf
    return 100 - (100 / (1 + rs))

def MACD(series, s_short=12, s_long=26, s_signal=9):
    ema_short = EMA(series, s_short)
    ema_long = EMA(series, s_long)
    macd = ema_short - ema_long
    signal = macd.ewm(span=s_signal, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def bollinger(series, window=20, stds=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + stds * std
    lower = ma - stds * std
    return upper, ma, lower


# -------------------------------------------------------
#              FETCHERS ( DATA PROVIDERS )
# -------------------------------------------------------

# ---- Binance (with your proxy) ----
@cache_ttl(ttl=60)
def fetch_ohlc_binance(symbol, interval="1h", limit=500):
    url = f"{BINANCE_PROXY}/api.binance.com/api/v3/klines"

    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)

    if r.status_code != 200:
        raise ValueError(f"Binance HTTP {r.status_code}: {r.text[:200]}")

    j = r.json()
    if not isinstance(j, list):
        raise ValueError("Invalid Binance response")

    rows = []
    for k in j:
        rows.append({
            "datetime": pd.to_datetime(k[0], unit='ms'),
            "Open": float(k[1]),
            "High": float(k[2]),
            "Low": float(k[3]),
            "Close": float(k[4]),
            "Volume": float(k[5])
        })

    df = pd.DataFrame(rows).set_index("datetime")
    return df


# ---- TwelveData ----
@cache_ttl(ttl=60)
def fetch_ohlc_twelvedata(symbol, interval, outputsize=500, timezone="UTC"):
    if not TWELVE_API:
        raise ValueError("TWELVEDATA_API_KEY not configured")

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "timezone": timezone,
        "format": "JSON",
        "apikey": TWELVE_API
    }
    r = requests.get(url, params=params, timeout=10)

    if r.status_code != 200:
        raise ValueError(f"TwelveData HTTP {r.status_code}")

    j = r.json()
    if "values" not in j:
        raise ValueError(f"TwelveData error: {j.get('message') or j}")

    df = pd.DataFrame(j["values"])
    df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}, inplace=True)

    df["Open"]   = df["Open"].astype(float)
    df["High"]   = df["High"].astype(float)
    df["Low"]    = df["Low"].astype(float)
    df["Close"]  = df["Close"].astype(float)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")

    return df


# ---- YahooQuery ----
@cache_ttl(ttl=60)
def fetch_ohlc_yahooquery(symbol, interval, period="7d"):
    t = Ticker(symbol)
    df = t.history(interval=interval, period=period)

    if df is None or df.empty:
        raise ValueError("No data from yahooquery")

    if isinstance(df, pd.DataFrame):
        data = df.copy()
    else:
        data = pd.DataFrame(df)

    data.columns = [c.capitalize() if c.lower() in ["open","high","low","close"] else c for c in data.columns]

    if "datetime" in data.columns:
        data["datetime"] = pd.to_datetime(data["datetime"])
        data = data.set_index("datetime")

    return data


# ---- Combined fetch selector ----
def fetch_ohlc(symbol, interval, limit_or_period=500, source="binance"):
    s = (source or "").lower()
    if s == "binance": return fetch_ohlc_binance(symbol, interval, limit_or_period)
    if s == "twelve": return fetch_ohlc_twelvedata(symbol, interval, limit_or_period)
    if s == "yahoo": return fetch_ohlc_yahooquery(symbol, interval, f"{max(1, int(limit_or_period/24))}d")
    raise ValueError("Unknown source")


# -------------------------------------------------------
#               NEWS + FOREX FACTORY
# -------------------------------------------------------

@cache_ttl(ttl=300)
def fetch_forexfactory(limit=30):
    url = "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.xml"
    try:
        feed = feedparser.parse(url)
        items = []
        for e in feed.entries[:limit]:
            items.append({
                "title": e.get("title", ""),
                "published": e.get("published", "")
            })
        return items
    except:
        return []


@cache_ttl(ttl=120)
def fetch_news_for_symbol(symbol, limit=10):
    headlines = []
    try:
        t = Ticker(symbol)
        j = t.news

        if isinstance(j, dict):
            for k in j:
                for it in j[k][:limit]:
                    title = it.get("title") or it.get("headline")
                    if title: headlines.append(title)

        elif isinstance(j, list):
            for it in j[:limit]:
                title = it.get("title") or it.get("headline")
                if title: headlines.append(title)

    except:
        pass

    # add forex factory too
    try:
        ff = fetch_forexfactory(10)
        for f in ff[:limit]:
            headlines.append(f.get("title",""))
    except:
        pass

    return headlines


# -------------------------------------------------------
#                 SIGNALS + BACKTEST
# -------------------------------------------------------

def generate_signals(df):
    df = df.copy()
    if df.empty: return df

    df["EMA12"] = EMA(df["Close"], 12)
    df["EMA26"] = EMA(df["Close"], 26)
    df["RSI14"] = RSI(df["Close"], 14)
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = MACD(df["Close"])
    df["BB_UP"], df["BB_MID"], df["BB_LOW"] = bollinger(df["Close"])

    df.dropna(inplace=True)

    # MACD cross
    df["MACD_cross"] = (df["MACD"] > df["MACD_SIGNAL"]).astype(int)

    df["signal"] = 0
    df.loc[(df["MACD_cross"].diff()==1) & (df["RSI14"]<75), "signal"] = 1
    df.loc[(df["MACD_cross"].diff()==-1) | (df["RSI14"]>85), "signal"] = -1

    return df


def simple_backtest(df, signal_col="signal"):
    trades = []
    position = None

    for i in range(len(df)-1):
        s = df.iloc[i][signal_col]
        next_open = float(df.iloc[i+1]["Open"])
        date = df.index[i+1]

        # ENTRY
        if s == 1 and position is None:
            position = {"entry_price": next_open, "entry_time": date}

        # EXIT
        elif s == -1 and position is not None:
            entry = position
            exit_price = next_open
            ret = (exit_price - entry["entry_price"]) / entry["entry_price"]

            trades.append({
                "entry_time": entry["entry_time"].isoformat(),
                "exit_time": date.isoformat(),
                "entry_price": float(entry["entry_price"]),
                "exit_price": float(exit_price),
                "return": float(ret)
            })

            position = None

    # close last open trade
    if position is not None:
        last_price = float(df.iloc[-1]["Close"])
        ret = (last_price - position["entry_price"]) / position["entry_price"]

        trades.append({
            "entry_time": position["entry_time"].isoformat(),
            "exit_time": df.index[-1].isoformat(),
            "entry_price": float(position["entry_price"]),
            "exit_price": float(last_price),
            "return": float(ret)
        })

    total_return = np.prod([1+t["return"] for t in trades]) - 1 if trades else 0
    avg_ret = np.mean([t["return"] for t in trades]) if trades else 0
    win_rate = np.mean([1 if t["return"]>0 else 0 for t in trades]) if trades else 0

    return trades, {
        "total_return": float(total_return),
        "avg_return": float(avg_ret),
        "win_rate": float(win_rate),
        "n_trades": len(trades)
    }


def hybrid_signal_score(df, headlines):
    last = df.iloc[-1]

    score = 0
    if last["EMA12"] > last["EMA26"]: score += 0.5
    else: score -= 0.5

    if last["MACD"] > last["MACD_SIGNAL"]: score += 0.3
    else: score -= 0.3

    if last["RSI14"] < 30: score += 0.2
    if last["RSI14"] > 70: score -= 0.2

    # news sentiment (simple)
    news_score = 0
    for h in headlines:
        h = h.lower()
        if "up" in h or "gain" in h or "surge" in h:
            news_score += 1
        if "down" in h or "fall" in h or "loss" in h:
            news_score -= 1

    news_score = max(-5, min(5, news_score)) / 5

    final = 0.7*score + 0.3*news_score
    reason = f"technical={score:.2f}, news={news_score:.2f}"

    return final, reason


# -------------------------------------------------------
#                     API ENDPOINTS
# -------------------------------------------------------

@app.get("/analyze")
async def analyze(symbol: str, interval: str="1h", limit: int=500, source: str="binance"):
    try:
        df = fetch_ohlc(symbol, interval, limit, source)

        df2 = generate_signals(df)
        if df2.empty:
            return JSONResponse({"error": "Not enough data"}, status_code=200)

        trades, metrics = simple_backtest(df2)
        headlines = fetch_news_for_symbol(symbol, 10)
        ff = fetch_forexfactory(10)

        hybrid_score, hybrid_reason = hybrid_signal_score(df2, headlines)

        candles = df2[["Open","High","Low","Close"]].tail(500).reset_index().to_dict("records")
        indicators = df2[["EMA12","EMA26","RSI14","MACD","MACD_SIGNAL","BB_UP","BB_LOW"]].tail(500).reset_index().to_dict("records")

        # متن تحلیل کامل‌تر:
        interp = [
            f"Hybrid Score = {hybrid_score:.3f}",
            f"Reason = {hybrid_reason}",
            f"Total Trades = {metrics['n_trades']}",
            f"Win Rate = {metrics['win_rate']*100:.1f}%",
        ]
        if trades:
            last_t = trades[-1]
            interp.append(
                f"Last Trade → Entry: {last_t['entry_price']}, Exit: {last_t['exit_price']}, Return: {last_t['return']*100:.2f}%"
            )

        return {
            "chart": candles,
            "indicators": indicators,
            "trades": trades,
            "metrics": metrics,
            "interpretation": interp,
            "headlines": headlines,
            "forexfactory": ff
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/symbols")
def symbols(source: str="binance"):
    if source=="binance":
        return ["BTCUSDT","ETHUSDT","DOGEUSDT","BNBUSDT","XRPUSDT","SOLUSDT"]
    if source=="twelve":
        return ["AAPL","TSLA","MSFT","EUR/USD","USD/JPY"]
    return ["BTC-USD","ETH-USD","AAPL","TSLA"]


@app.get("/health")
def health():
    return {"status":"ok"}


# -------- static folder --------
app.mount("/", StaticFiles(directory="static", html=True), name="static")
