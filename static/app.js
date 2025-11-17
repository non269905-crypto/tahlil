let chart, candleSeries, ema12Series, ema26Series;

function initChart() {
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
    ema12Series = chart.addLineSeries({ lineWidth: 1, priceLineVisible: false });
    ema26Series = chart.addLineSeries({ lineWidth: 1, priceLineVisible: false });

    window.addEventListener("resize", () =>
        chart.applyOptions({ width: container.clientWidth })
    );
}

function loadSymbols() {
    fetch("symbols.json")
        .then(r => r.json())
        .then(list => {
            const box = document.getElementById("symbols-list");
            list.forEach(s => {
                const opt = document.createElement("option");
                opt.value = s;
                box.appendChild(opt);
            });
        });
}

function prepareTime(t) {
    let d = new Date(t);
    if (isNaN(d)) return (new Date(t)).toISOString();
    return d.toISOString();
}

async function loadData() {
    const symbol = document.getElementById("symbol").value || "BTCUSDT";
    const interval = document.getElementById("interval").value;
    const source = document.getElementById("source").value;

    document.getElementById("summary").innerText = "در حال بارگذاری...";

    const res = await fetch(`/analyze?symbol=${symbol}&interval=${interval}&source=${source}`);
    const data = await res.json();
    
    if (data.error) {
        alert("Error: " + data.error);
        return;
    }

    // chart
    const c = data.chart.map(o => ({
        time: prepareTime(o.datetime),
        open: +o.Open,
        high: +o.High,
        low: +o.Low,
        close: +o.Close
    }));
    candleSeries.setData(c);

    // indicators
    const i = data.indicators;
    const ema12 = i.map((it, idx) => ({ time: c[idx].time, value: it.EMA12 }));
    const ema26 = i.map((it, idx) => ({ time: c[idx].time, value: it.EMA26 }));

    ema12Series.setData(ema12);
    ema26Series.setData(ema26);

    // trades
    const trades = data.trades || [];
    candleSeries.setMarkers(
        trades.flatMap(t => [
            {
                time: prepareTime(t.entry_time),
                position: "belowBar",
                color: t.return > 0 ? "green" : "red",
                shape: "arrowUp",
                text: "Entry"
            },
            {
                time: prepareTime(t.exit_time),
                position: "aboveBar",
                color: t.return > 0 ? "green" : "red",
                shape: "arrowDown",
                text: "Exit"
            }
        ])
    );

    document.getElementById("summary").innerText = trades.length
        ? trades.map(t => `ورود ${t.entry_price} خروج ${t.exit_price} (${(t.return * 100).toFixed(2)}%)`).join(" | ")
        : "بدون معامله";

    document.getElementById("metrics").innerText = JSON.stringify(data.metrics, null, 2);

    document.getElementById("news").innerText = (data.headlines || []).join("\n\n---\n\n");
    document.getElementById("ff").innerText = (data.forexfactory || []).map(f => `${f.published}: ${f.title}`).join("\n\n---\n\n");
}

initChart();
loadSymbols();
document.getElementById("btn").addEventListener("click", loadData);
