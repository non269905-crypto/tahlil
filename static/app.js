let chart, candleSeries, ema12Series, ema26Series;

function initChart() {
    const container = document.getElementById("chart");
    if (!container) return;
    chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: 560,
        layout: { textColor: "#e6eef8", background: { color: "#071221" } },
        grid: { vertLines: { color: "#0b2230" }, horzLines: { color: "#0b2230" } },
        rightPriceScale: { borderColor: "#0b2230" },
        timeScale: { borderColor: "#0b2230" }
    });
    candleSeries = chart.addCandlestickSeries({
        upColor: "#26a69a", downColor: "#ef5350",
        borderUpColor: "#26a69a", borderDownColor: "#ef5350",
        wickUpColor: "#26a69a", wickDownColor: "#ef5350"
    });
    ema12Series = chart.addLineSeries({ color: "#00bcd4", lineWidth: 2 });
    ema26Series = chart.addLineSeries({ color: "#ff9800", lineWidth: 2 });

    window.addEventListener("resize", () => {
        if (chart) {
            chart.applyOptions({ width: container.clientWidth });
        }
    });
}

function toUnixSeconds(t) {
    const d = new Date(t);
    if (isNaN(d) && typeof t === 'number') {
        return Math.floor(t / 1000);
    }
    return Math.floor(d.getTime() / 1000);
}

// *** رفع خطای ۴۰۴: بارگذاری نمادها و سپس داده‌ها ***
async function loadSymbolsAndData() {
    const source = document.getElementById("source").value;
    const sel = document.getElementById("symbol");
    
    document.getElementById("summary").innerText = `در حال بارگذاری نمادها از ${source}...`;

    // 1. Load symbols based on current source
    try {
        const res = await fetch(`/symbols?source=${source}`);
        const syms = await res.json();
        
        sel.innerHTML = "";
        syms.forEach(s => {
            const o = document.createElement("option");
            o.value = s;
            o.textContent = s;
            sel.appendChild(o);
        });

        // 2. Set a default symbol
        if (syms.length > 0) {
             if (source === "yahoo") {
                sel.value = syms.find(s => s === "BTC-USD") || syms[0];
             } else {
                sel.value = syms[0]; 
             }
        }

    } catch (e) {
        console.error("Symbols load error:", e);
        sel.innerHTML = "<option value=''>خطا در بارگذاری نمادها</option>";
        document.getElementById("summary").innerText = "خطا: نمادها بارگذاری نشدند.";
        return; 
    }

    // 3. If symbols loaded successfully, load the data
    if (sel.value) {
        loadData();
    }
}

async function loadData() {
    const symbol = document.getElementById("symbol").value;
    const interval = document.getElementById("interval").value || "1h";
    const source = document.getElementById("source").value || "twelve";

    if (!symbol) {
        document.getElementById("summary").innerText = "لطفا یک نماد انتخاب کنید.";
        return;
    }

    document.getElementById("summary").innerText = "در حال بارگذاری داده‌های معاملاتی...";
    document.getElementById("metrics").innerText = "";
    document.getElementById("interpret").innerText = "";
    document.getElementById("news").innerText = "—";
    document.getElementById("news_fa").innerText = "—";
    document.getElementById("ff").innerText = "—";
    candleSeries.setData([]);
    ema12Series.setData([]);
    ema26Series.setData([]);

    try {
        const res = await fetch(`/analyze?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}&source=${encodeURIComponent(source)}&limit=500`);
        const j = await res.json();
        
        if (j.error) {
            alert(`خطا در ${source}: ${j.error}`);
            document.getElementById("summary").innerText = `خطا از ${source}: ${j.error}`;
            return;
        }

        // Show chart data
        const chartData = (j.chart || []).map(r => ({
            time: toUnixSeconds(r.datetime),
            open: +r.Open,
            high: +r.High,
            low: +r.Low,
            close: +r.Close
        }));
        
        candleSeries.setData(chartData);
        if(chart) chart.timeScale().fitContent();

        // Show indicators 
        const inds = j.indicators || [];
        const ema12 = inds.map(it=>({ time: toUnixSeconds(it.datetime), value: it.EMA12 })).filter(x=>x.value!==null);
        const ema26 = inds.map(it=>({ time: toUnixSeconds(it.datetime), value: it.EMA26 })).filter(x=>x.value!==null);
        
        ema12Series.setData(ema12);
        ema26Series.setData(ema26);

        // Show trades and metrics
        const trades = j.trades || [];
        candleSeries.setMarkers(trades.flatMap(t=>[
            { time: toUnixSeconds(t.entry_time), position: "belowBar", color: t.return>0 ? "green":"red", shape:"arrowUp", text:"Entry"},
            { time: toUnixSeconds(t.exit_time), position: "aboveBar", color: t.return>0 ? "green":"red", shape:"arrowDown", text:"Exit"}
        ]));

        document.getElementById("metrics").innerText = JSON.stringify(j.metrics || {}, null, 2);
        document.getElementById("summary").innerText = trades.length ? `${trades.length} معامله` : "بدون معامله";
        
        // Show analysis
        document.getElementById("interpret").innerText = (j.interpretation || []).join("\n");
        document.getElementById("news").innerText = (j.headlines || []).join("\n\n---\n\n");
        document.getElementById("news_fa").innerText = (j.translated_headlines || []).join("\n\n---\n\n");
        document.getElementById("ff").innerText = (j.forexfactory || []).map(f=> `${f.published.split('T')[0]} — ${f.title}`).join("\n\n---\n\n");

    } catch (e) {
        console.error("Data load error:", e);
        alert("خطای ارتباطی: " + e.message);
        document.getElementById("summary").innerText = "خطا در بارگذاری داده‌ها";
    }
}

// *** Event Listeners ***
document.addEventListener("DOMContentLoaded", () => {
    initChart();
    loadSymbolsAndData(); 

    document.getElementById("btn").addEventListener("click", loadData);
    
    document.getElementById("source").addEventListener("change", loadSymbolsAndData);

    document.getElementById("symbol").addEventListener("change", loadData);
    document.getElementById("interval").addEventListener("change", loadData);
});
