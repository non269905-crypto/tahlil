# main.py
import os
import time
import math
import requests
import numpy as np
import pandas as pd
from functools import wraps
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from yahooquery import Ticker

TWELVE_API = os.getenv("TWELVEDATA_API_KEY", "").strip()
if not TWELVE_API:
    print("Warning: TWELVEDATA_API_KEY not set; TwelveData requests will fail if used.")

app = FastAPI(title="Hybrid Trading Analysis")

# Simple in-memory cache with TTL
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

# --- Indicators ---
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

# --- Fetch OHLC from TwelveData (primary) ---
@cache_ttl(ttl=60)  # cache 60s
def fetch_ohlc_twelvedata(symbol, interval, outputsize=500, timezone="UTC"):
    """
    interval examples: 1min, 5min, 15min, 1h, 1day
    """
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
        # possible error: {"status":"error","message":"..."}
        raise ValueError(f"TwelveData error: {j.get('message') or j}")
    vals = j["values"]
    df = pd.DataFrame(vals).astype(float)
    # TwelveData returns newest first
    df = df.sort_values("datetime").reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.index = pd.to_datetime(df.index)
    df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}, inplace=True)
    return df

# --- Fallback fetch using yahooquery (for tickers not available on TwelveData or news)
@cache_ttl(ttl=60)
def fetch_ohlc_yahooquery(symbol, interval, period='7d'):
    # yahooquery Ticker.history supports intervals like '1h', '1d', '1m'
    t = Ticker(symbol)
    df = t.history(interval=interval, period=period)
    if df is None or df.empty:
        raise ValueError("No data from yahooquery")
    # history may have multiindex; normalize
    if isinstance(df, pd.DataFrame):
        local = df.copy()
    else:
        local = pd.DataFrame(df)
    # some yahooquery returns columns ['open','high','low','close','volume']
    local = local.rename(columns=lambda c: c.capitalize() if c in ['open','high','low','close','volume'] else c)
    if 'datetime' in local.columns:
        local['datetime'] = pd.to_datetime(local['datetime'])
        local.set_index('datetime', inplace=True)
    # Ensure OHLC present
    if not set(['Open','High','Low','Close']).issubset(set(local.columns)):
        raise ValueError("yahooquery missing ohlc")
    return local

# --- Simple sentiment from headlines ---
POS = {"up", "rise", "positive", "beat", "strong", "gain", "higher", "hawkish", "cut", "eased", "easing", "falling"}
NEG = {"down", "drop", "miss", "weak", "loss", "lower", "bear", "bearish", "hike", "higher-than-expected", "surge"}
def headline_sentiment(headline):
    h = str(headline).lower()
    score = 0
    for w in POS:
        if w in h: score += 1
    for w in NEG:
        if w in h: score -= 1
    return score

@cache_ttl(ttl=120)
def fetch_news_for_symbol(symbol, limit=10):
    # first try yahooquery
    try:
        t = Ticker(symbol)
        j = t.news
        if isinstance(j, dict) and symbol in j:
            items = j[symbol][:limit]
        elif isinstance(j, list):
            items = j[:limit]
        else:
            items = []
        headlines = [it.get('title') or it.get('headline') for it in items if it]
        return headlines
    except Exception:
        return []

# --- Signal generation (technical) ---
def generate_signals(df):
    df = df.copy()
    if df.empty:
        return df
    df['EMA12'] = EMA(df['Close'], 12)
    df['EMA26'] = EMA(df['Close'], 26)
    df['RSI14'] = RSI(df['Close'], 14)
    df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = MACD(df['Close'])
    df['BB_UP'], df['BB_MID'], df['BB_LOW'] = bollinger(df['Close'], window=20)
    df.dropna(subset=['EMA26','RSI14','MACD_SIGNAL'], inplace=True)
    if df.empty:
        return df
    df['MACD_cross'] = (df['MACD'] > df['MACD_SIGNAL']).astype(int)
    df['signal'] = 0
    df.loc[(df['MACD_cross'].diff() == 1) & (df['RSI14'] < 75), 'signal'] = 1
    df.loc[(df['MACD_cross'].diff() == -1) | (df['RSI14'] > 85), 'signal'] = -1
    df['signal'].fillna(0, inplace=True)
    return df

# --- Backtest & trades (entry at next open, exit at next open) ---
def simple_backtest(df, signal_col='signal'):
    trades = []
    position = None
    if df.empty:
        return trades, {'total_return':0,'avg_return':0,'win_rate':0,'n_trades':0}
    for i in range(len(df)-1):
        s = df.iloc[i][signal_col]
        next_open = float(df.iloc[i+1]['Open'])
        date = df.index[i+1]
        if s == 1 and position is None:
            position = {'entry_price': next_open, 'entry_time': date}
        elif s == -1 and position is not None:
            entry = position
            exit_price = next_open
            ret = (exit_price - entry['entry_price']) / entry['entry_price']
            trades.append({
                'entry_time': entry['entry_time'].isoformat(),
                'exit_time': date.isoformat(),
                'entry_price': float(entry['entry_price']),
                'exit_price': float(exit_price),
                'return': float(ret)
            })
            position = None
    if position is not None:
        last_price = float(df.iloc[-1]['Close'])
        ret = (last_price - position['entry_price']) / position['entry_price']
        trades.append({
            'entry_time': position['entry_time'].isoformat(),
            'exit_time': df.index[-1].isoformat(),
            'entry_price': float(position['entry_price']),
            'exit_price': float(last_price),
            'return': float(ret)
        })
    total_return = np.prod([1 + t['return'] for t in trades]) - 1 if trades else 0.0
    avg_ret = np.mean([t['return'] for t in trades]) if trades else 0.0
    win_rate = np.mean([1 if t['return']>0 else 0 for t in trades]) if trades else 0.0
    return trades, {'total_return':float(total_return),'avg_return':float(avg_ret),'win_rate':float(win_rate),'n_trades':len(trades)}

# --- Combine news + technical for weighted signal ---
def hybrid_signal_score(df, headlines):
    # compute last row technical momentum score
    if df.empty:
        return 0.0, "No data"
    last = df.iloc[-1]
    score = 0.0
    # technical pieces
    if last['EMA12'] > last['EMA26']:
        score += 0.5
    else:
        score -= 0.5
    if last['MACD'] > last['MACD_SIGNAL']:
        score += 0.3
    else:
        score -= 0.3
    if last['RSI14'] < 30:
        score += 0.2
    if last['RSI14'] > 70:
        score -= 0.2
    # news sentiment
    s = 0
    for h in headlines:
        s += headline_sentiment(h)
    # normalize news to [-1,1]
    news_score = max(-3, min(3, s)) / 3.0 if headlines else 0.0
    # weight: technical 0.7, news 0.3
    final = 0.7 * score + 0.3 * news_score
    reason = f"tech={score:.2f},news={news_score:.2f}"
    return final, reason

# --- API endpoints ---
@app.get("/analyze")
async def analyze(symbol: str = Query("AAPL"), interval: str = Query("1h"), source: str = Query("twelve"), period:int = Query(500)):
    """
    symbol: ticker symbol (TwelveData or Yahoo format)
    interval: 1min,5min,15min,30min,1h,4h,1day
    source: 'twelve' or 'yahoo' (preferred)
    period: outputsize (how many candles)
    """
    try:
        df = None
        last_exception = None
        # try TwelveData first if requested
        if source == 'twelve':
            try:
                df = fetch_ohlc_twelvedata(symbol, interval, outputsize=period)
            except Exception as e:
                last_exception = e
                # fallback to yahooquery
                try:
                    df = fetch_ohlc_yahooquery(symbol, interval, period=f"{max(1,int(period/24))}d")
                except Exception as e2:
                    raise ValueError(f"TwelveData failed: {e}; yahoo fallback failed: {e2}")
        else:
            # yahoo primary
            df = fetch_ohlc_yahooquery(symbol, interval, period=f"{max(1,int(period/24))}d")

        if df is None or df.empty:
            raise ValueError("No data returned")

        df2 = generate_signals(df)
        if df2.empty:
            return JSONResponse({
                "chart": [], "indicators": [], "trades": [], "metrics": {'total_return':0,'avg_return':0,'win_rate':0,'n_trades':0},
                "interpretation":["Not enough rows after indicator cleaning"]
            }, status_code=200)

        trades, metrics = simple_backtest(df2)
        headlines = fetch_news_for_symbol(symbol, limit=6)
        hybrid_score, hybrid_reason = hybrid_signal_score(df2, headlines)

        # prepare chart payload (tail last N candles)
        tail = df2[['Open','High','Low','Close']].tail(500).reset_index()
        tail['index'] = tail['datetime'].astype(str) if 'datetime' in tail.columns else tail['datetime'].astype(str)
        chart = tail.to_dict(orient='records')

        indicators = df2[['EMA12','EMA26','RSI14','MACD','MACD_SIGNAL','BB_UP','BB_LOW']].tail(500).reset_index().to_dict(orient='records')

        return {
            "chart": chart,
            "indicators": indicators,
            "trades": trades,
            "metrics": metrics,
            "interpretation": [
                f"Hybrid score: {hybrid_score:.3f} ({hybrid_reason})",
            ],
            "headlines": headlines
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
def health():
    return {"status":"ok"}

# mount static
app.mount("/", StaticFiles(directory="static", html=True), name="static")
