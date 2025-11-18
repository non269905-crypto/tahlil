# main.py
import os, time, requests
from functools import wraps
from typing import List
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from yahooquery import Ticker
import feedparser

# -----------------------------------------
#  SETTINGS
# -----------------------------------------

# آدرس پروکسی Cloudflare Worker شما 
BINANCE_PROXY = "https://pro2.bagheryane.workers.dev" 

TWELVE_API = os.getenv("TWELVEDATA_API_KEY", "").strip()

app = FastAPI(title="Hybrid Trading Analysis - MultiSource")

# serve static
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")


# ---------- simple in-memory cache (TTL) ----------
_cache = {}
def cache_ttl(ttl=30):
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            
            # FIX: تبدیل لیست‌ها و مجموعه‌ها به Tuple برای قابل هش شدن (رفع خطای unhashable type: 'list')
            
            # 1. مدیریت آرگومان‌های موقعیتی (*args)
            hashable_args = []
            for arg in args:
                if isinstance(arg, (list, set)):
                    hashable_args.append(tuple(arg))
                else:
                    hashable_args.append(arg)
            
            # 2. مدیریت آرگومان‌های کلمه‌ای (**kwargs)
            hashable_kwargs = tuple(
                sorted(
                    (k, tuple(v) if isinstance(v, (list, set)) else v) 
                    for k, v in kwargs.items()
                )
            )
            
            key = (func.__name__, tuple(hashable_args), hashable_kwargs)
            
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

# ---------- indicators ----------
def EMA(s, span): return s.ewm(span=span, adjust=False).mean()
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

# ---------- fetchers ----------
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
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        raise ValueError(f"TwelveData HTTP {r.status_code}: {r.text[:200]}")
    j = r.json()
    if "values" not in j:
        raise ValueError(f"TwelveData error: {j}")
    vals = j["values"]
    df = pd.DataFrame(vals)
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values("datetime").reset_index(drop=True)
    df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}, inplace=True)
    df.set_index('datetime', inplace=True)
    return df

@cache_ttl(ttl=60)
def fetch_ohlc_binance(symbol, interval="1h", limit=500):
    base = BINANCE_PROXY.rstrip("/")
    url = f"{base}/api/v3/klines"
    
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=15)
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

def map_interval_to_yahoo(interval: str) -> str:
    """Maps app interval names (e.g., '1min') to yahooquery's required format (e.g., '1m')."""
    mapping = {
        "1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m",
        "1h": "1h", "4h": "90m",
        "1day": "1d",
    }
    return mapping.get(interval, "1h")

@cache_ttl(ttl=60)
def fetch_ohlc_yahooquery(symbol, interval, period='7d'):
    t = Ticker(symbol)
    df = t.history(interval=interval, period=period)
    
    if df is None or df.empty:
        raise ValueError("No data from yahooquery (possibly symbol/interval/period mismatch)")
        
    if isinstance(df, dict):
        df = pd.DataFrame(df)
        
    local = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    local.columns = [c.capitalize() if c.lower() in ['open','high','low','close','volume'] else c for c in local.columns]
    
    # Logic to handle index normalization (to fix date cndate errors)
    if 'Datetime' in local.columns:
        local['datetime'] = pd.to_datetime(local['Datetime'])
        local.set_index('datetime', inplace=True)
    elif 'date' in local.columns:
        local['datetime'] = pd.to_datetime(local['Date'])
        local.set_index('datetime', inplace=True)
    elif local.index.name.lower() in ['date','datetime']:
        local.index = pd.to_datetime(local.index)
        local.index.name = 'datetime'
    else:
        local.index = pd.to_datetime(local.index)
        local.index.name = 'datetime'

    if not set(['Open','High','Low','Close']).issubset(set(local.columns)):
        raise ValueError("yahooquery missing ohlc")
        
    return local[['Open','High','Low','Close']].copy()

def fetch_ohlc(symbol, interval, limit_or_period=500, source="twelve"):
    source = (source or "twelve").lower()
    if source == "binance":
        return fetch_ohlc_binance(symbol, interval, limit_or_period)
    if source == "twelve":
        return fetch_ohlc_twelvedata(symbol, interval, outputsize=limit_or_period)
    if source == "yahoo":
        yahoo_interval = map_interval_to_yahoo(interval)
        period_str = f"{max(1,int(limit_or_period/24))}d"
        if yahoo_interval in ["1m", "5m", "15m", "30m"]:
            period_str = "7d"
            
        return fetch_ohlc_yahooquery(symbol, yahoo_interval, period=period_str)
        
    raise ValueError("Unknown source")

# ---------- ForexFactory RSS ----------
@cache_ttl(ttl=300)
def fetch_forexfactory(limit=30):
    url = "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.xml"
    try:
        feed = feedparser.parse(url)
        items = []
        for e in feed.entries[:limit]:
            items.append({"title": e.get("title",""), "published": e.get("published",""), "raw": str(e)})
        return items
    except Exception:
        return []

# ---------- simple news / headlines ----------
@cache_ttl(ttl=120)
def fetch_news_for_symbol(symbol, limit=10):
    headlines = []
    try:
        t = Ticker(symbol)
        j = t.news
        if isinstance(j, dict):
            for k in j:
                items = j.get(k, [])
                for it in items[:limit]:
                    if it:
                        headlines.append(it.get("title") or it.get("headline") or "")
        elif isinstance(j, list):
            for it in j[:limit]:
                headlines.append(it.get("title") or it.get("headline") or "")
    except Exception:
        pass
    try:
        ff = fetch_forexfactory(limit=5)
        for f in ff[:limit]:
            headlines.append(f.get("title",""))
    except:
        pass
    return [h for h in headlines if h]

# ---------- translation using LibreTranslate ----------
@cache_ttl(ttl=300)
def translate_texts_to_fa(texts: List[str]):
    if not texts:
        return []
    out = []
    url = "https://translate.argosopentech.com/translate"
    for t in texts:
        try:
            r = requests.post(url, json={"q": t, "source": "auto", "target": "fa", "format": "text"}, timeout=8)
            if r.status_code == 200:
                j = r.json()
                out.append(j.get("translatedText", t))
            else:
                out.append(t)
        except Exception:
            out.append(t)
    return out

# ---------- simple sentiment ----------
POS = {"up","rise","positive","beat","strong","gain","higher","cut","eased","easing","surge"}
NEG = {"down","drop","miss","weak","loss","lower","bear","bearish","hike","inflation"}
def headline_sentiment(headline):
    h = str(headline).lower()
    s = 0
    for w in POS:
        if w in h: s += 1
    for w in NEG:
        if w in h: s -= 1
    return s

# ---------- signals / backtest ----------
def generate_signals(df):
    df = df.copy()
    df['EMA12'] = EMA(df['Close'], 12)
    df['EMA26'] = EMA(df['Close'], 26)
    df['RSI14'] = RSI(df['Close'], 14)
    df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = MACD(df['Close'])
    df['BB_UP'], df['BB_MID'], df['BB_LOW'] = bollinger(df['Close'])
    df.dropna(subset=['EMA26','RSI14','MACD_SIGNAL'], inplace=True)
    if df.empty:
        return df
    df['MACD_cross'] = (df['MACD'] > df['MACD_SIGNAL']).astype(int)
    df['signal'] = 0
    df.loc[(df['MACD_cross'].diff() == 1) & (df['RSI14'] < 75), 'signal'] = 1
    df.loc[(df['MACD_cross'].diff() == -1) | (df['RSI14'] > 85), 'signal'] = -1
    df['signal'].fillna(0, inplace=True)
    return df

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
    avg_ret = float(np.mean([t['return'] for t in trades]) if trades else 0.0)
    win_rate = float(np.mean([1 if t['return']>0 else 0 for t in trades]) if trades else 0.0)
    max_win = float(max([t['return'] for t in trades]) if trades else 0.0)
    max_loss = float(min([t['return'] for t in trades]) if trades else 0.0)
    return trades, {'total_return':total_return,'avg_return':avg_ret,'win_rate':win_rate,'n_trades':len(trades),'max_win':max_win,'max_loss':max_loss}

# ---------- hybrid scoring ----------
def hybrid_signal_score(df, headlines: List[str]):
    if df.empty:
        return 0.0, "No data"
    last = df.iloc[-1]
    score = 0.0
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
    s = 0
    for h in headlines:
        s += headline_sentiment(h)
    news_score = max(-5, min(5, s)) / 5.0 if headlines else 0.0
    final = 0.7 * score + 0.3 * news_score
    reason = f"tech={score:.2f},news={news_score:.2f}"
    return final, reason

# ---------- endpoints ----------
@app.get("/symbols")
def symbols(source: str = Query("binance")):
    if source == "binance":
        try:
            r = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=10)
            j = r.json()
            syms = [s["symbol"] for s in j.get("symbols", []) if s.get("status")=="TRADING"]
            top = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT"]
            uniq = top + [s for s in syms if s not in top]
            return uniq[:600]
        except Exception:
            # Fallback when Binance API is inaccessible
            return ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT"]
    if source == "twelve":
        return ["AAPL","TSLA","MSFT","GOOGL","EURUSD","GBPUSD","XAUUSD"]
    if source == "yahoo":
        return ["BTC-USD", "ETH-USD", "AAPL", "TSLA", "^GSPC"]
    return []

@app.get("/analyze")
def analyze(symbol: str = Query("BTCUSDT"), interval: str = Query("1h"), limit: int = Query(500), source: str = Query("binance")):
    try:
        df = fetch_ohlc(symbol, interval, limit, source)
        if df is None or df.empty:
            raise ValueError("No data returned")
        df2 = generate_signals(df)
        if df2 is None or df2.empty:
            return JSONResponse({"chart":[],"indicators":[],"trades":[],"metrics":{},"interpretation":["not enough data"],"headlines":[],"forexfactory":[],"translated_headlines":[]}, status_code=200)
        
        trades, metrics = simple_backtest(df2)
        headlines = fetch_news_for_symbol(symbol, limit=8)
        translated = translate_texts_to_fa(headlines)
        ff = fetch_forexfactory(limit=8)
        hybrid_score, hybrid_reason = hybrid_signal_score(df2, headlines)
        
        tail = df2[['Open','High','Low','Close']].tail(limit).reset_index()
        tail.rename(columns={"index":"datetime"}, inplace=True)
        indicators = df2[['EMA12','EMA26','RSI14','MACD','MACD_SIGNAL','BB_UP','BB_LOW']].tail(limit).reset_index().to_dict(orient='records')
        
        interpretation = [
            f"Hybrid score: {hybrid_score:.3f} ({hybrid_reason})",
            f"Signals count: {int(metrics.get('n_trades',0))}"
        ]
        
        return {
            "chart": tail.to_dict(orient='records'),
            "indicators": indicators,
            "trades": trades,
            "metrics": metrics,
            "interpretation": interpretation,
            "headlines": headlines,
            "translated_headlines": translated,
            "forexfactory": ff
        }
    except Exception as e:
        # Custom handlers for known TwelveData errors
        if "TwelveData HTTP" in str(e) and "401" in str(e):
            return JSONResponse({"error": "TwelveData API Key is invalid or expired."}, status_code=500)
        if "TwelveData HTTP" in str(e) and "429" in str(e):
            return JSONResponse({"error": "TwelveData rate limit exceeded."}, status_code=500)
            
        # The final, generic JSON error response
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
def health():
    return {"status":"ok"}
