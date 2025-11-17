# main.py
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time

app = FastAPI(title="Trading Analytics Advanced")

# --- Indicators ---
def EMA(s, span): return s.ewm(span=span, adjust=False).mean()
def SMA(s, window): return s.rolling(window=window).mean()

def RSI(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # اصلاح: استفاده از ewm برای RSI استانداردتر
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

# --- Signals and simple backtest ---
def generate_signals(df):
    df = df.copy().dropna()
    df['EMA12'] = EMA(df['Close'], 12)
    df['EMA26'] = EMA(df['Close'], 26)
    df['RSI14'] = RSI(df['Close'], 14)
    df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = MACD(df['Close'])
    df['BB_UP'], df['BB_MID'], df['BB_LOW'] = bollinger(df['Close'])

    # MACD crossover signal and RSI filter
    df['MACD_cross'] = (df['MACD'] > df['MACD_SIGNAL']).astype(int)
    df['signal'] = 0
    # buy: MACD crosses above AND RSI not extreme
    df.loc[(df['MACD_cross'].diff() == 1) & (df['RSI14'] < 75), 'signal'] = 1
    # sell: MACD crosses below OR RSI extremely high
    df.loc[(df['MACD_cross'].diff() == -1) | (df['RSI14'] > 85), 'signal'] = -1

    return df

def simple_backtest(df, signal_col='signal'):
    trades = []
    position = None
    for i in range(len(df)-1):
        s = df.iloc[i][signal_col]
        if i + 1 >= len(df):
            break
            
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
    return trades, {'total_return': float(total_return), 'avg_return': float(avg_ret), 'win_rate': float(win_rate), 'n_trades': len(trades)}

# --- Fetch OHLC ---
def fetch_ohlc(symbol, interval, period='7d'):
    df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
    if df is None or df.empty:
        raise ValueError("No data returned for symbol/interval")
    df.index = pd.to_datetime(df.index)
    return df

# --- Interpretations ---
def interpret(df):
    if df.empty:
        return ["No data for interpretation."]
        
    last = df.iloc[-1]
    texts = []
    
    # ✅ اصلاح برای رفع خطای Ambiguous: استفاده از pd.notna() قبل از مقایسه
    if pd.notna(last['EMA12']) and pd.notna(last['EMA26']):
        if last['EMA12'] > last['EMA26']:
            texts.append("EMA12 above EMA26 → short-term bullish trend")
        else:
            texts.append("EMA12 below EMA26 → short-term bearish trend")
            
    if pd.notna(last['RSI14']):
        if last['RSI14'] < 30:
            texts.append("RSI indicates oversold conditions (possible rebound)")
        elif last['RSI14'] > 70:
            texts.append("RSI indicates overbought conditions (possible pullback)")
            
    if pd.notna(last['Close']) and pd.notna(last['BB_UP']) and pd.notna(last['BB_LOW']):
        if last['Close'] > last['BB_UP']:
            texts.append("Price above upper Bollinger band (extended move)")
        elif last['Close'] < last['BB_LOW']:
            texts.append("Price below lower Bollinger band (extended move)")
            
    if not texts:
        texts.append("Not enough data for interpretation.")
        
    return texts

# --- API endpoints ---

# ✅ اصلاح: استفاده از async def
@app.get("/analyze")
async def analyze(symbol: str = Query("GC=F"), interval: str = Query("5m"), period: str = Query("7d")):
    try:
        df = fetch_ohlc(symbol, interval, period=period)
        df_signals = generate_signals(df)
        trades, metrics = simple_backtest(df_signals)
        interp = interpret(df_signals)
        
        chart_df = df_signals[['Open','High','Low','Close']].tail(200).reset_index()
        chart_df['index'] = chart_df['index'].astype(str)
        chart = chart_df.to_dict(orient='records')
        
        indicators_df = df_signals[['EMA12','EMA26','RSI14','MACD','MACD_SIGNAL','BB_UP','BB_LOW']].tail(200).reset_index()
        indicators_df['index'] = indicators_df['index'].astype(str)
        indicators = indicators_df.to_dict(orient='records')

        return {"chart": chart, "indicators": indicators, "trades": trades, "metrics": metrics, "interpretation": interp}
    except Exception as e:
        # در صورت بروز خطای Ambiguous، اینجا یک پاسخ 500 با پیام خطا برگردانده می‌شود
        return JSONResponse({"error": str(e)}, status_code=500)

# ✅ اصلاح: استفاده از async def
@app.get("/news")
async def news():
    sources = [
        "https://www.forexfactory.com/ffcal_week_this.xml",
        "https://finance.yahoo.com/rss/topstories"
    ]
    items = []
    for s in sources:
        try:
            r = requests.get(s, timeout=5)
            if r.status_code == 200:
                items.append(r.text[:2000])
        except requests.RequestException:
            continue
            
    return {"news": items, "count": len(items), "time": time.time()}


# ✅ اصلاح: انتقال app.mount به انتهای فایل برای رفع ارور 404
app.mount("/", StaticFiles(directory="static", html=True), name="static")
