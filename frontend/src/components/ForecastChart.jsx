import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";

export default function ForecastChart({ actual, predicted, quantiles = {} }) {
  const [pointIndex, setPointIndex] = useState(0);
  const [lineIndex, setLineIndex] = useState(0);
  const [phase, setPhase] = useState("points");

  useEffect(() => {
    setPointIndex(0);
    setLineIndex(0);
    setPhase("points");

    const pointInterval = setInterval(() => {
      setPointIndex((prev) => {
        if (prev < actual.length) return prev + 1;
        clearInterval(pointInterval);
        setPhase("lines");
        return prev;
      });
    }, 150);

    return () => clearInterval(pointInterval);
  }, [actual, predicted]);

  useEffect(() => {
    if (phase !== "lines") return;

    const lineInterval = setInterval(() => {
      setLineIndex((prev) => {
        if (prev < actual.length) return prev + 1;
        clearInterval(lineInterval);
        return prev;
      });
    }, 100);

    return () => clearInterval(lineInterval);
  }, [phase]);

  const xPoints = [...Array(pointIndex).keys()];
  const xLines = [...Array(lineIndex).keys()];
  const residuals = actual.map((v, i) => v - predicted[i]);

  return (
    <div>
      {/* === Forecast Chart === */}
      <h2>1. Forecast Plot: Actual vs. Predicted (Animated)</h2>
      <Plot
        data={[
          {
            x: xPoints,
            y: actual.slice(0, pointIndex),
            type: "scatter",
            mode: "markers",
            name: "Actual Points",
            marker: { color: "#1f77b4", size: 6 }
          },
          {
            x: xPoints,
            y: predicted.slice(0, pointIndex),
            type: "scatter",
            mode: "markers",
            name: "Predicted Points",
            marker: { color: "#ff7f0e", size: 6 }
          },
          {
            x: xLines,
            y: actual.slice(0, lineIndex),
            type: "scatter",
            mode: "lines",
            name: "Actual Line",
            line: { color: "#1f77b4", width: 2 }
          },
          {
            x: xLines,
            y: predicted.slice(0, lineIndex),
            type: "scatter",
            mode: "lines",
            name: "Predicted Line",
            line: { color: "#ff7f0e", width: 2, dash: "dash" }
          }
        ]}
        layout={{
          title: "Forecast (Animated)",
          xaxis: { title: "Time", range: [0, actual.length] },
          yaxis: { title: "Value" },
          margin: { t: 50, l: 50, r: 30, b: 50 }
        }}
        config={{ displayModeBar: false }}
      />

      {/* === Residual Plot === */}
      <h2>2. Residual Plot: Actual - Predicted</h2>
      <Plot
        data={[
          {
            x: [...Array(actual.length).keys()],
            y: residuals,
            type: "bar",
            name: "Residuals",
            marker: { color: "#6b7280" }
          }
        ]}
        layout={{
          title: "Residuals",
          xaxis: { title: "Time" },
          yaxis: { title: "Residual (Actual - Predicted)" },
          margin: { t: 50, l: 50, r: 30, b: 50 }
        }}
        config={{ displayModeBar: false }}
      />

      {/* === Quantile Plot === */}
      {Object.keys(quantiles).length > 0 && (
  <>
    <h2>3. Quantile Prediction Plot</h2>
    <Plot
      data={(() => {
        const levels = Object.keys(quantiles)
          .map(Number)
          .sort((a, b) => a - b);

        const bands = [];
        for (let i = 0; i < levels.length - 1; i++) {
          const lower = levels[i];
          const upper = levels[i + 1];

          bands.push({
            x: [...Array(quantiles[lower].length).keys()],
            y: quantiles[upper],
            fill: 'tonexty',
            fillcolor: `rgba(0, 123, 255, ${0.1 + i * 0.1})`,
            line: { color: 'transparent' },
            name: `${lower}% - ${upper}%`,
            type: 'scatter'
          });

          bands.push({
            x: [...Array(quantiles[lower].length).keys()],
            y: quantiles[lower],
            line: { color: 'transparent' },
            name: '',
            type: 'scatter'
          });
        }

        // Add the median (50%) line on top
        bands.push({
          x: [...Array(quantiles[50].length).keys()],
          y: quantiles[50],
          type: 'scatter',
          mode: 'lines',
          name: 'Median (50%)',
          line: { color: '#000', width: 2 }
        });

        return bands;
      })()}
      layout={{
        title: "Prediction Intervals by Quantile",
        xaxis: { title: "Time" },
        yaxis: { title: "Forecast Value" },
        margin: { t: 50, l: 50, r: 30, b: 50 },
        showlegend: true
      }}
      config={{ displayModeBar: false }}
    />
  </>
)}

    </div>
  );
}
