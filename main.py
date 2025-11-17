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

TWELVE_API = os.getenv("TWELVEDATA_API_KEY", "").strip()

app = FastAPI(title="Hybrid Trading Analysis - MultiSource")

# Serve UI
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Cache
_cache = {}
def cache_ttl(ttl=30):
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            k = (func.__name__, args, tuple(sorted(kwargs.items())))
            now = time.time()
            if k in _cache:
                v, exp = _cache[k]
                if now < exp:
                    return v
            v = func(*args, **kwargs)
            _cache[k] = (v, now + ttl)
            return v
        return wrapper
    return deco

# Indicators
def EMA(s, span): return s.ewm(span=span, adjust=False).mean()
def RSI(series, period=14):
    d = series.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ma_up = up.ewm(com=period - 1, adjust=True, min_periods=period).mean()
    ma_dn = dn.ewm(com=period - 1, adjust=True, min_periods=period).mean()
    rs = ma_up / ma_dn
    rs[ma_dn == 0] = np.inf
    return 100 - (100 / (1 + rs))
def MACD(series, a=12, b=26, c=9):
    es = EMA(series, a)
    el = EMA(series, b)
    m = es - el
    sig = m.ewm(span=c, adjust=False).mean()
    return m, sig, m - sig
def bollinger(series, w=20, s=2):
    ma = series.rolling(w).mean()
    sd = series.rolling(w).std()
    return ma + s*sd, ma, ma - s*sd

# Fetchers
@cache_ttl(60)
def fetch_ohlc_twelvedata(symbol, interval, outputsize=500):
    if not TWELVE_API:
        raise ValueError("TWELVEDATA_API_KEY missing")
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_API
    }
    r = requests.get(url, params=params, timeout=10)
    j = r.json()
    if "values" not in j:
        raise ValueError(str(j))
    df = pd.DataFrame(j["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.astype({"open":float,"high":float,"low":float,"close":float,"volume":float})
    df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close"}, inplace=True)
    df.set_index("datetime", inplace=True)
    return df.sort_index()

@cache_ttl(60)
def fetch_ohlc_binance(symbol, interval="1h", limit=500):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol":symbol.upper(),"interval":interval,"limit":limit}
    r = requests.get(url, params=params, timeout=10)
    j = r.json()
    rows = [{
        "datetime":pd.to_datetime(k[0],unit='ms'),
        "Open":float(k[1]),
        "High":float(k[2]),
        "Low":float(k[3]),
        "Close":float(k[4]),
        "Volume":float(k[5])
    } for k in j]
    df = pd.DataFrame(rows).set_index("datetime")
    return df

@cache_ttl(60)
def fetch_ohlc_yahoo(symbol, interval="1h", days=7):
    t = Ticker(symbol)
    df = t.history(interval=interval, period=f"{days}d")
    if df is None or df.empty:
        raise ValueError("Yahoo returned empty")
    df = df.rename(columns=str.capitalize)
    df.index = pd.to_datetime(df.index)
    return df[["Open","High","Low","Close","Volume"]]

def fetch_ohlc(symbol, interval, limit, source):
    source = source.lower()
    if source=="binance": return fetch_ohlc_binance(symbol, interval, limit)
    if source=="twelve": return fetch_ohlc_twelvedata(symbol, interval, limit)
    if source=="yahoo": return fetch_ohlc_yahoo(symbol, interval, days=7)
    raise ValueError("Unknown source")

# News
@cache_ttl(120)
def fetch_news(symbol, limit=10):
    out=[]
    try:
        t = Ticker(symbol)
        d=t.news
        if isinstance(d,list):
            for i in d[:limit]:
                out.append(i.get("title",""))
    except: pass

    try:
        ff = fetch_forexfactory(limit=5)
        for x in ff:
            out.append(x["title"])
    except: pass

    return out

@cache_ttl(300)
def fetch_forexfactory(limit=20):
    feed = feedparser.parse("https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.xml")
    return [{"title":e.get("title",""),"published":e.get("published","")} for e in feed.entries[:limit]]

# Indicators + Backtest
def generate_signals(df):
    df["EMA12"]=EMA(df["Close"],12)
    df["EMA26"]=EMA(df["Close"],26)
    df["RSI14"]=RSI(df["Close"],14)
    df["MACD"],df["MACD_SIGNAL"],df["MACD_HIST"]=MACD(df["Close"])
    df["BB_UP"],_,df["BB_LOW"]=bollinger(df["Close"])
    df.dropna(inplace=True)

    df["MACD_cross"] = (df["MACD"]>df["MACD_SIGNAL"]).astype(int)
    df["signal"]=0
    df.loc[(df["MACD_cross"].diff()==1)&(df["RSI14"]<75),"signal"]=1
    df.loc[(df["MACD_cross"].diff()==-1)| (df["RSI14"]>85),"signal"]=-1
    return df

def simple_backtest(df):
    trades=[]
    pos=None
    for i in range(len(df)-1):
        s=df.iloc[i]["signal"]
        nxt=float(df.iloc[i+1]["Open"])
        ts=df.index[i+1]
        if s==1 and pos is None:
            pos={"entry":nxt,"time":ts}
        elif s==-1 and pos:
            entry=pos
            ret=(nxt-entry["entry"])/entry["entry"]
            trades.append({"entry_time":entry["time"].isoformat(),
                           "exit_time":ts.isoformat(),
                           "entry_price":entry["entry"],
                           "exit_price":nxt,
                           "return":ret})
            pos=None

    metrics={"n_trades":len(trades),
             "win_rate":float(np.mean([1 if t["return"]>0 else 0 for t in trades]) if trades else 0),
             "avg_return":float(np.mean([t["return"] for t in trades]) if trades else 0),
             "total_return":float(np.prod([1+t["return"] for t in trades])-1 if trades else 0)}
    return trades,metrics

# Endpoint
@app.get("/analyze")
def analyze(symbol:str, interval:str="1h", limit:int=500, source:str="binance"):
    try:
        df=fetch_ohlc(symbol, interval, limit, source)
        df=generate_signals(df)
        trades,metrics = simple_backtest(df)
        news = fetch_news(symbol, limit=10)
        ff = fetch_forexfactory(limit=10)

        chart = df[["Open","High","Low","Close"]].tail(limit).reset_index().to_dict("records")
        inds = df[["EMA12","EMA26","RSI14","MACD","MACD_SIGNAL","BB_UP","BB_LOW"]].tail(limit).reset_index().to_dict("records")

        return {
            "chart":chart,
            "indicators":inds,
            "trades":trades,
            "metrics":metrics,
            "headlines":news,
            "forexfactory":ff,
        }
    except Exception as e:
        return JSONResponse({"error":str(e)},status_code=500)

@app.get("/health")
def health():
    return {"status":"ok"}
