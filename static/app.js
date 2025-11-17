let chart, candleSeries, ema12Series, ema26Series;

function initChart() {
    const container = document.getElementById("chart");
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
        chart.applyOptions({ width: container.clientWidth });
    });
}

function toUnixSeconds(t) {
    // accept ISO string or pandas timestamp
    const d = new Date(t);
    if (isNaN(d)) return Math.floor(Date.now() / 1000);
    return Math.floor(d.getTime() / 1000);
}

async function loadSymbols() {
    try {
        const res = await fetch("/symbols?source=binance");
        const syms = await res.json();
        const sel = document.getElementById("symbol");
        sel.innerHTML = "";
        syms.slice(0,200).forEach(s => {
            const o = document.createElement("option");
            o.value = s;
            o.textContent = s;
            sel.appendChild(o);
        });
        // set default
        if (!sel.value) sel.value = "BTCUSDT";
    } catch (e) {
        console.error("symbols load error", e);
        const sel = document.getElementById("symbol");
        sel.innerHTML = "<option>BTCUSDT</option>";
    }
}

async function loadData() {
    const symbol = document.getElementById("symbol").value || "BTCUSDT";
    const interval = document.getElementById("interval").value || "1h";
    const source = document.getElementById("source").value || "binance";

    document.getElementById("summary").innerText = "در حال بارگذاری...";

    try {
        const res = await fetch(`/analyze?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}&source=${encodeURIComponent(source)}&limit=500`);
        const j = await res.json();
        if (j.error) {
            alert("خطا: " + j.error);
            document.getElementById("summary").innerText = "خطا";
            return;
        }

        const chartData = (j.chart || []).map(r => ({
            time: toUnixSeconds(r.datetime),
            open: +r.Open,
            high: +r.High,
            low: +r.Low,
            close: +r.Close
        }));
        candleSeries.setData(chartData);

        const inds = j.indicators || [];
        const ema12 = inds.map((it,i)=>({ time: chartData[i]?.time, value: it.EMA12 })).filter(x=>x.value!==null);
        const ema26 = inds.map((it,i)=>({ time: chartData[i]?.time, value: it.EMA26 })).filter(x=>x.value!==null);
        ema12Series.setData(ema12);
        ema26Series.setData(ema26);

        const trades = j.trades || [];
        candleSeries.setMarkers(trades.flatMap(t=>[
            { time: toUnixSeconds(t.entry_time), position: "belowBar", color: t.return>0 ? "green":"red", shape:"arrowUp", text:"Entry"},
            { time: toUnixSeconds(t.exit_time), position: "aboveBar", color: t.return>0 ? "green":"red", shape:"arrowDown", text:"Exit"}
        ]));

        document.getElementById("metrics").innerText = JSON.stringify(j.metrics || {}, null, 2);
        document.getElementById("summary").innerText = trades.length ? `${trades.length} معامله` : "بدون معامله";

        document.getElementById("interpret").innerText = (j.interpretation || []).join("\n");
        document.getElementById("news").innerText = (j.headlines || []).join("\n\n---\n\n");
        document.getElementById("news_fa").innerText = (j.translated_headlines || []).join("\n\n---\n\n");
        document.getElementById("ff").innerText = (j.forexfactory || []).map(f=> `${f.published} — ${f.title}`).join("\n\n---\n\n");

    } catch (e) {
        console.error(e);
        alert("خطا در بارگذاری: " + e);
        document.getElementById("summary").innerText = "خطا";
    }
}

initChart();
loadSymbols();
document.getElementById("btn").addEventListener("click", loadData);
