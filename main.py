import os, time, requests
from functools import wraps
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from yahooquery import Ticker
import feedparser

# -----------------------------------------
#  SETTINGS
# -----------------------------------------

# Your Cloudflare Worker proxy
BINANCE_PROXY = "https://pro.bagheryane.workers.dev"

# TwelveData API key (optional)
TWELVE_API = os.getenv("TWELVEDATA_API_KEY", "").strip()

app = FastAPI(title="Hybrid Trading Server")


# -----------------------------------------
#  SIMPLE TTL CACHE
# -----------------------------------------

_cache = {}
def cache(ttl=30):
    def wrap(func):
        def run(*args, **kwargs):
            key = (func.__name__, args, tuple(sorted(kwargs.items())))
            now = time.time()
            if key in _cache:
                val, exp = _cache[key]
                if now < exp:
                    return val
            val = func(*args, **kwargs)
            _cache[key] = (val, now + ttl)
            return val
        return run
    return wrap


# -----------------------------------------
#  INDICATORS
# -----------------------------------------

def EMA(s, p): return s.ewm(span=p, adjust=False).mean()

def RSI(s, p=14):
    d = s.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    ma_up = up.ewm(com=p - 1, adjust=True).mean()
    ma_dn = dn.ewm(com=p - 1, adjust=True).mean()
    rs = ma_up / ma_dn
    return 100 - (100 / (1 + rs))

def MACD(s):
    e12, e26 = EMA(s, 12), EMA(s, 26)
    macd = e12 - e26
    signal = macd.ewm(span=9).mean()
    return macd, signal


# -----------------------------------------
#  FETCH Binance (via Cloudflare Worker Proxy)
# -----------------------------------------

@cache(60)
def fetch_binance(symbol: str, interval: str, limit: int):
    base = BINANCE_PROXY.rstrip("/")
    url = f"{base}/api/v3/klines"             # correct path for Worker
    
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit
    }

    r = requests.get(url, params=params, timeout=10)

    if r.status_code != 200:
        raise ValueError(f"Binance HTTP {r.status_code}: {r.text[:500]}")

    j = r.json()
    if not isinstance(j, list):
        raise ValueError("Invalid Binance response")

    rows = []
    for c in j:
        rows.append({
            "datetime": pd.to_datetime(c[0], unit="ms"),
            "Open": float(c[1]),
            "High": float(c[2]),
            "Low": float(c[3]),
            "Close": float(c[4]),
            "Volume": float(c[5]),
        })

    df = pd.DataFrame(rows).set_index("datetime")
    return df


# -----------------------------------------
#  FETCH TwelveData
# -----------------------------------------

@cache(60)
def fetch_twelvedata(symbol, interval, limit):
    if not TWELVE_API:
        raise ValueError("TWELVEDATA_API_KEY is missing")

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": limit,
        "apikey": TWELVE_API
    }

    r = requests.get(url, params=params, timeout=10)
    j = r.json()

    vals = j.get("values", [])
    if not vals:
        raise ValueError(f"TwelveData error: {j}")

    df = pd.DataFrame(vals)
    df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")
    df = df.astype(float)
    return df


# -----------------------------------------
#  FETCH YahooQuery
# -----------------------------------------

@cache(60)
def fetch_yahoo(symbol, interval, limit):
    t = Ticker(symbol)
    df = t.history(interval=interval, period=f"{max(1,limit//24)}d")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("YahooQuery returned no data")

    df.rename(columns=lambda x: x.capitalize(), inplace=True)
    df["datetime"] = pd.to_datetime(df["date"])
    df = df.set_index("datetime")
    return df[["Open","High","Low","Close"]]


# -----------------------------------------
#  DISPATCH FETCHER
# -----------------------------------------

def fetch_data(symbol, interval, limit, source):
    if source == "binance":
        return fetch_binance(symbol, interval, limit)
    if source == "twelve":
        return fetch_twelvedata(symbol, interval, limit)
    if source == "yahoo":
        return fetch_yahoo(symbol, interval, limit)
    raise ValueError("Invalid source")


# -----------------------------------------
#  NEWS + FOREXFACTORY
# -----------------------------------------

@cache(180)
def fetch_news(symbol):
    titles = []

    # yahooquery news
    try:
        t = Ticker(symbol)
        n = t.news or []
        for item in n[:10]:
            titles.append(item.get("title") or "")
    except:
        pass

    # forex factory
    try:
        ff = feedparser.parse("https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.xml")
        for e in ff.entries[:10]:
            titles.append(e.get("title",""))
    except:
        pass

    return [t for t in titles if t]


# -----------------------------------------
#  SIGNALS + BACKTEST
# -----------------------------------------

def compute_signals(df):
    df = df.copy()
    df["EMA12"] = EMA(df.Close, 12)
    df["EMA26"] = EMA(df.Close, 26)
    df["RSI"]   = RSI(df.Close)
    df["MACD"], df["MACD_SIGNAL"] = MACD(df.Close)
    df.dropna(inplace=True)

    df["sig"] = 0
    df.loc[(df.MACD > df.MACD_SIGNAL) & (df.RSI < 70), "sig"] = 1
    df.loc[(df.MACD < df.MACD_SIGNAL) | (df.RSI > 85), "sig"] = -1
    return df


def backtest(df):
    trades = []
    pos = None

    for i in range(len(df)-1):
        sig = df.iloc[i]["sig"]
        next_open = df.iloc[i+1]["Open"]
        next_time = df.index[i+1]

        if sig == 1 and pos is None:
            pos = {"entry": next_open, "time": next_time}

        elif sig == -1 and pos:
            ret = (next_open - pos["entry"]) / pos["entry"]
            trades.append({
                "entry_price": pos["entry"],
                "exit_price": next_open,
                "return": ret,
                "entry_time": pos["time"].isoformat(),
                "exit_time": next_time.isoformat(),
            })
            pos = None

    metrics = {
        "n": len(trades),
        "avg": float(np.mean([t["return"] for t in trades])) if trades else 0,
        "win": float(np.mean([t["return"] > 0 for t in trades])) if trades else 0
    }

    return trades, metrics


# -----------------------------------------
#  API: /analyze
# -----------------------------------------

@app.get("/analyze")
def analyze(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("1h"),
    limit: int = Query(500),
    source: str = Query("binance")
):
    try:
        df = fetch_data(symbol, interval, limit, source)
        df2 = compute_signals(df)
        trades, metrics = backtest(df2)
        headlines = fetch_news(symbol)

        chart = df2[["Open","High","Low","Close"]].reset_index().to_dict("records")
        indicators = df2[["EMA12","EMA26","RSI","MACD","MACD_SIGNAL"]].reset_index().to_dict("records")

        return {
            "chart": chart,
            "indicators": indicators,
            "trades": trades,
            "metrics": metrics,
            "news": headlines
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# -----------------------------------------
#  /symbols
# -----------------------------------------

@app.get("/symbols")
def symbols(source: str = "binance"):
    if source == "binance":
        return ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT"]
    if source == "twelve":
        return ["AAPL","TSLA","MSFT","AMZN"]
    return ["BTC-USD","ETH-USD","AAPL"]


# -----------------------------------------
#  STATIC FILES
# -----------------------------------------

app.mount("/", StaticFiles(directory="static", html=True), name="static")
