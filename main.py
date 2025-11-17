<!doctype html>
<html lang="fa">
<head>
<meta charset="utf-8"/>
<title>Hybrid Trading Dashboard (MultiSource)</title>
<script src="https://unpkg.com/lightweight-charts@3.9.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
body{font-family:tahoma, sans-serif;background:#071221;color:#e6eef8;padding:12px;direction:rtl;}
.container{max-width:1200px;margin:auto;}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px;}
.select, input, button{padding:8px;border-radius:6px;background:#0b1a2a;border:1px solid #173147;color:#e6eef8;}
#chart{height:520px;background:#071221;border-radius:6px;padding:6px;}
.panel{margin-top:12px;background:#081826;padding:10px;border-radius:6px;}
.small{font-size:0.9rem;color:#9fb0c8;}
.btn{background:#1f8cff;border:none;padding:8px 12px;border-radius:6px;color:#fff;cursor:pointer}
</style>
</head>
<body>
<div class="container">
<h2>داشبورد ترکیبی — Binance / TwelveData / Yahoo + ForexFactory</h2>
<div class="controls">
<label>نماد:
<input id="symbol" class="select" value="BTCUSDT" />
</label>
<label>منبع:
<select id="source" class="select">
  <option value="binance">Binance (Crypto)</option>
  <option value="twelve">TwelveData (Stocks/FX)</option>
  <option value="yahoo">YahooQuery (Fallback)</option>
</select>
</label>
<label>تایمفریم:
<select id="interval" class="select">
<option>1min</option><option>5min</option><option>15min</option><option>30min</option><option selected>1h</option><option>4h</option><option>1day</option>
</select>
</label>
<button id="btn" class="btn">بارگذاری</button>
</div>

<div id="chart"></div>

<div class="panel">
<strong>خلاصه معاملات:</strong> <span id="summary">-</span><br>
<strong>متریک‌ها:</strong> <pre id="metrics" class="small"></pre>
</div>

<div class="panel">
<h4>تحلیل ترکیبی (تکنیکال + اخبار)</h4>
<div id="interpret"></div>
</div>

<div class="panel">
<h4>اخبار (Yahoo + ForexFactory)</h4>
<pre id="news" class="small">—</pre>
</div>

<div class="panel">
<h4>تقویم اقتصادی (ForexFactory)</h4>
<pre id="ff" class="small">—</pre>
</div>
</div>

<script>
const API_BASE = "";

let chart, candleSeries, ema12Series, ema26Series;

function initChart(){
    const container = document.getElementById('chart');
    chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: 520,
        layout: { background: { color: '#071221' }, textColor: '#e6eef8' },
        grid: { vertLines: { color: '#0b2230' }, horzLines: { color: '#0b2230' } },
        rightPriceScale: { borderColor: '#0b2230' },
        timeScale: { borderColor: '#0b2230' }
    });
    candleSeries = chart.addCandlestickSeries();
    ema12Series = chart.addLineSeries({lineWidth:1, priceLineVisible:false});
    ema26Series = chart.addLineSeries({lineWidth:1, priceLineVisible:false});
    window.addEventListener('resize', ()=> chart.applyOptions({width:container.clientWidth}));
}

function prepareTime(tstr){
    const d = new Date(tstr);
    if (isNaN(d)) {
        // try parse as UTC string
        return (new Date(tstr)).toISOString();
    }
    return d.toISOString();
}

async function loadData(){
    const symbol = document.getElementById('symbol').value;
    const interval = document.getElementById('interval').value;
    const source = document.getElementById('source').value;
    document.getElementById('summary').innerText = 'در حال بارگذاری...';
    try{
        const res = await fetch(`/analyze?symbol=${encodeURIComponent(symbol)}&interval=${interval}&source=${source}&limit=500`);
        const j = await res.json();
        if(j.error){ alert('Error: '+j.error); document.getElementById('summary').innerText='خطا'; return; }
        if(!j.chart || !j.chart.length){ alert('No data'); document.getElementById('summary').innerText='داده‌ای نیست'; return; }

        const cdata = j.chart.map(r=>({
            time: prepareTime(r.datetime || r.index || r.date),
            open: parseFloat(r.Open),
            high: parseFloat(r.High),
            low: parseFloat(r.Low),
            close: parseFloat(r.Close)
        }));
        candleSeries.setData(cdata);

        // indicators alignment
        const inds = j.indicators || [];
        const ema12 = inds.map((it, idx)=>({ time: cdata[idx] ? cdata[idx].time : prepareTime(it.datetime || it.index), value: it['EMA12'] })).filter(x=>x.value!=null);
        const ema26 = inds.map((it, idx)=>({ time: cdata[idx] ? cdata[idx].time : prepareTime(it.datetime || it.index), value: it['EMA26'] })).filter(x=>x.value!=null);
        ema12Series.setData(ema12);
        ema26Series.setData(ema26);

        // trades markers
        const markers = [];
        (j.trades||[]).forEach(t=>{
            markers.push({ time: prepareTime(t.entry_time), position: 'belowBar', color: t.return>0 ? 'green' : 'red', shape:'arrowUp', text: 'Entry' });
            markers.push({ time: prepareTime(t.exit_time), position: 'aboveBar', color: t.return>0 ? 'green' : 'red', shape:'arrowDown', text: 'Exit' });
        });
        candleSeries.setMarkers(markers);

        document.getElementById('metrics').innerText = JSON.stringify(j.metrics || {}, null, 2);
        document.getElementById('summary').innerText = (j.trades && j.trades.length) ? j.trades.map(t=>`Entry:${t.entry_price.toFixed(6)} Exit:${t.exit_price.toFixed(6)} Ret:${(t.return*100).toFixed(2)}%`).join(' | ') : 'هیچ معامله‌ای';
        document.getElementById('interpret').innerText = (j.interpretation||[]).join('\n');
        document.getElementById('news').innerText = (j.headlines && j.headlines.length) ? j.headlines.join('\n\n---\n\n') : 'خبری نیست';
        // ForexFactory block
        if(j.forexfactory && j.forexfactory.length){
            document.getElementById('ff').innerText = j.forexfactory.map(x=>`${x.published} — ${x.title}`).join('\n\n---\n\n');
        } else {
            document.getElementById('ff').innerText = 'تازه‌ترین اخبار تقویم اقتصادی در دسترس نیست';
        }
    }catch(e){
        console.error(e);
        alert('خطا در بارگذاری: '+e);
    }
}

initChart();
document.getElementById('btn').addEventListener('click', loadData);
</script>
</body>
</html>
