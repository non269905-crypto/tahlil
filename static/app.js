let chart, candleSeries, ema12Series, ema26Series;

function initChart() {

    const container = document.getElementById("chart");

    chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: 540,
        layout: {
            textColor: "#e6eef8",
            background: { color: "#071221" }
        },
        grid: {
            vertLines: { color: "#0b2230" },
            horzLines: { color: "#0b2230" }
        },
        rightPriceScale: { borderColor: "#0b2230" },
        timeScale: { borderColor: "#0b2230" }
    });

    candleSeries = chart.addCandlestickSeries({
        upColor: "#26a69a",
        downColor: "#ef5350",
        borderUpColor: "#26a69a",
        borderDownColor: "#ef5350",
        wickUpColor: "#26a69a",
        wickDownColor: "#ef5350"
    });

    ema12Series = chart.addLineSeries({
        color: "#00bcd4",
        lineWidth: 2
    });

    ema26Series = chart.addLineSeries({
        color: "#ff9800",
        lineWidth: 2
    });

    window.addEventListener("resize", () => {
        chart.applyOptions({width: container.clientWidth});
    });
}


function prepareTime(t){
    return new Date(t).getTime() / 1000;
}

async function loadData(){
    const symbol = document.getElementById("symbol").value;
    const interval = document.getElementById("interval").value;
    const source = document.getElementById("source").value;

    document.getElementById("summary").innerText = "در حال بارگذاری...";

    try{
        const res = await fetch(`/analyze?symbol=${symbol}&interval=${interval}&source=${source}&limit=500`);
        const j = await res.json();

        if(j.error){
            alert(j.error);
            return;
        }

        // ---- CHART DATA ----
        const ohlc = j.chart.map(c => ({
            time: prepareTime(c.datetime),
            open: c.Open,
            high: c.High,
            low: c.Low,
            close: c.Close
        }));

        candleSeries.setData(ohlc);

        // ---- INDICATORS ----
        const ema12 = j.indicators.map((c,i)=>({
            time: ohlc[i].time,
            value: c.EMA12
        })).filter(x=>x.value !== null);

        const ema26 = j.indicators.map((c,i)=>({
            time: ohlc[i].time,
            value: c.EMA26
        })).filter(x=>x.value !== null);

        ema12Series.setData(ema12);
        ema26Series.setData(ema26);

        // ---- TRADES ----
        const markers = [];
        (j.trades || []).forEach(t => {
            markers.push({
                time: prepareTime(t.entry_time),
                position: "belowBar",
                shape: "arrowUp",
                color: t.return > 0 ? "green" : "red"
            });

            markers.push({
                time: prepareTime(t.exit_time),
                position: "aboveBar",
                shape: "arrowDown",
                color: t.return > 0 ? "green" : "red"
            });
        });

        candleSeries.setMarkers(markers);

        // ---- METRICS ----
        document.getElementById("metrics").innerText =
            JSON.stringify(j.metrics, null, 2);

        // ---- SUMMARY ----
        document.getElementById("summary").innerText =
            j.trades.length
                ? j.trades.map(t => `Entry ${t.entry_price} → Exit ${t.exit_price} (Ret ${(t.return*100).toFixed(2)}%)`).join(" | ")
                : "هیچ معامله‌ای";

        // ---- INTERPRET ----
        document.getElementById("interpret").innerText =
            (j.interpretation || []).join("\n");

        // ---- NEWS ----
        document.getElementById("news").innerText =
            j.headlines.length ? j.headlines.join("\n\n---\n\n") : "خبری نیست";

        // ---- FOREX FACTORY ----
        document.getElementById("ff").innerText =
            j.forexfactory.length
                ? j.forexfactory.map(f=> `${f.published} — ${f.title}`).join("\n\n---\n\n")
                : "تقویم اقتصادی در دسترس نیست";

    }catch(e){
        alert("خطا: "+e);
    }
}

initChart();
document.getElementById("btn").addEventListener("click", loadData);
