let chart, candleSeries, ema12Series, ema26Series;

function initChart() {
    const container = document.getElementById('chart');

    chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: 520,
        layout: {
            background: { color: '#071221' },
            textColor: '#e6eef8'
        },
        grid: {
            vertLines: { color: '#0b2230' },
            horzLines: { color: '#0b2230' }
        }
    });

    candleSeries = chart.addCandlestickSeries();
    ema12Series = chart.addLineSeries({ color: "#00ccff", lineWidth: 1 });
    ema26Series = chart.addLineSeries({ color: "#ffaa00", lineWidth: 1 });

    window.addEventListener("resize", () =>
        chart.applyOptions({ width: container.clientWidth })
    );
}

function iso(t) { return new Date(t).toISOString(); }

async function loadSymbols() {
    try {
        const res = await fetch("/symbols?source=binance");
        const syms = await res.json();

        const sel = document.getElementById("symbol");
        sel.innerHTML = "";

        syms.forEach(s => {
            const o = document.createElement("option");
            o.value = s;
            o.textContent = s;
            sel.appendChild(o);
        });

    } catch (e) {
        console.error("Symbol load error", e);
    }
}

async function loadData() {
    const symbol = document.getElementById("symbol").value;
    const interval = document.getElementById("interval").value;
    const source = document.getElementById("source").value;

    document.getElementById("summary").innerText = "در حال بارگذاری...";

    try {
        const res = await fetch(`/analyze?symbol=${symbol}&interval=${interval}&source=${source}&limit=500`);
        const j = await res.json();

        if (!j.chart) {
            alert("خطا در دریافت اطلاعات");
            return;
        }

        // chart
        const cdata = j.chart.map(r => ({
            time: iso(r.datetime),
            open: r.Open,
            high: r.High,
            low: r.Low,
            close: r.Close
        }));
        candleSeries.setData(cdata);

        // indicators
        const ema12 = j.indicators.map((r, i) => ({
            time: cdata[i]?.time,
            value: r.EMA12
        }));
        const ema26 = j.indicators.map((r, i) => ({
            time: cdata[i]?.time,
            value: r.EMA26
        }));

        ema12Series.setData(ema12);
        ema26Series.setData(ema26);

        // trades
        const markers = [];
        j.trades.forEach(t => {
            markers.push({
                time: iso(t.entry_time),
                position: 'belowBar',
                shape: 'arrowUp',
                color: t.return > 0 ? 'green' : 'red'
            });
            markers.push({
                time: iso(t.exit_time),
                position: 'aboveBar',
                shape: 'arrowDown',
                color: t.return > 0 ? 'green' : 'red'
            });
        });
        candleSeries.setMarkers(markers);

        // UI info
        document.getElementById("summary").innerText =
            j.trades.length
                ? j.trades.map(t => `Entry:${t.entry_price} Exit:${t.exit_price} Ret:${(t.return * 100).toFixed(2)}%`).join(" | ")
                : "هیچ معامله‌ای نبود";

        document.getElementById("metrics").innerText = JSON.stringify(j.metrics, null, 2);
        document.getElementById("interpret").innerText = j.interpretation.join("\n");
        document.getElementById("news").innerText = j.headlines.join("\n\n---\n");
        document.getElementById("ff").innerText = j.forexfactory.map(x => `${x.published} — ${x.title}`).join("\n\n");
    }
    catch (e) {
        alert("خطا: " + e);
        console.error(e);
    }
}

initChart();
loadSymbols();
document.getElementById("btn").addEventListener("click", loadData);
