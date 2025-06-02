import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";

export default function ForecastChart({ forecastData, historicalData }) {
  const [forecastProgress, setForecastProgress] = useState({});
  
  useEffect(() => {
    if (!forecastData || forecastData.length === 0) return;

    const entities = [...new Set(forecastData.map(item => item.Entity))];
    const timers = [];

    entities.forEach(entity => {
      const entityForecast = forecastData.filter(item => item.Entity === entity);
      let index = 0;

      const interval = setInterval(() => {
        setForecastProgress(prev => ({
          ...prev,
          [entity]: index + 1
        }));
        index++;
        if (index >= entityForecast.length) clearInterval(interval);
      }, 200); // Change speed (ms) as needed

      timers.push(interval);
    });

    return () => timers.forEach(clearInterval);
  }, [forecastData]);

  if (!forecastData || forecastData.length === 0) {
    return <p>Forecast Data not available..</p>;
  }

  const entities = [...new Set(forecastData.map(item => item.Entity))];

  return (
    <div>
      {entities.map(entity => {
        const entityForecast = forecastData.filter(item => item.Entity === entity);
        const histEntry = historicalData?.data.find(d => d.entity === entity);
        const entityHistory = histEntry ? histEntry.values : [];

        const quantileKeys = Object.keys(entityForecast[0]).filter(key =>
          key.startsWith("Quantile_")
        );

        const sortedQuantiles = quantileKeys.sort((a, b) => {
          const getNum = q => parseFloat(q.replace("Quantile_", ""));
          return getNum(a) - getNum(b);
        });

        const progress = forecastProgress[entity] || 0;
        const currentForecast = entityForecast.slice(0, progress);

        const traces = [];

        // Historical data (full line)
        if (entityHistory.length > 0) {
          traces.push({
            x: entityHistory.map(item => item.Date),
            y: entityHistory.map(item => item.Value),
            mode: "lines",
            name: "Historical Data",
            line: { color: "blue", width: 2, dash: "solid" }
          });
        }

        // Forecast: 10-90 band (light reddish-orange)
        const q10 = sortedQuantiles.find(q => q.includes("10"));
        const q90 = sortedQuantiles.find(q => q.includes("90"));
        if (q10 && q90 && currentForecast.length > 0) {
          traces.push({
            x: [...currentForecast.map(d => d.Date), ...currentForecast.map(d => d.Date).reverse()],
            y: [...currentForecast.map(d => d[q90]), ...currentForecast.map(d => d[q10]).reverse()],
            fill: "toself",
            fillcolor: "rgba(255, 71, 71, 0.3)",
            line: { color: "transparent" },
            name: "10th-90th Percentile",
            type: "scatter",
            mode: "lines",
            showlegend: true
          });
        }

        // Forecast: 25-75 band (darker reddish-orange)
        const q25 = sortedQuantiles.find(q => q.includes("25"));
        const q75 = sortedQuantiles.find(q => q.includes("75"));
        if (q25 && q75 && currentForecast.length > 0) {
          traces.push({
            x: [...currentForecast.map(d => d.Date), ...currentForecast.map(d => d.Date).reverse()],
            y: [...currentForecast.map(d => d[q75]), ...currentForecast.map(d => d[q25]).reverse()],
            fill: "toself",
            fillcolor: "rgba(255, 71, 71, 0.4)",
            line: { color: "transparent" },
            name: "25th-75th Percentile",
            type: "scatter",
            mode: "lines",
            showlegend: true
          });
        }

        // Forecast: Median (50th) dashed line
        const q50 = sortedQuantiles.find(q => q.includes("50"));
        if (q50 && currentForecast.length > 0) {
          traces.push({
            x: currentForecast.map(d => d.Date),
            y: currentForecast.map(d => d[q50]),
            mode: "lines+markers",
            name: "Median (50th Percentile)",
            line: { color: "rgb(255, 71, 71)", width: 2, dash: "dash" }
          });
        }

        return (
          <div key={entity} style={{ marginBottom: "2rem" }}>
            <h2 style={{ color: "darkblue" }}>Forecast for {entity}</h2>
            <Plot
              data={traces}
              layout={{
                title: `Forecast & Historical Data for ${entity}`,
                xaxis: { title: "Date" },
                yaxis: { title: "Value" },
                margin: { t: 50, l: 50, r: 30, b: 50 },
                showlegend: true
              }}
              config={{ displayModeBar: false }}
            />
          </div>
          
        );
      })}
    </div>
  );
}
