import React, { useState } from "react";
import Header from "./components/Header";
// import DatasetSelector from "./components/DatasetSelector";
// import ModelSelector from "./components/ModelSelector";
import ForecastChart from "./components/ForecastChart";
import MetricsDisplay from "./components/MetricsDisplay";
import './App.css';



function App() {
  // const [file, setFile] = useState(null);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [metadata, setMetadata] = useState({});
  const [errorMetric,setErrorMetric] = useState({});
  const [forecastData, setForecastData] = useState([]);
  const [historicalData, setHistoricalData] = useState([]);
  const [isLoading, setIsLoading] = useState(false);


  const datasets =  [
    "covid_death",
    "covid_confirmed",
    "covid_recovered",
    "african_GDP",
    "co2",
    "air_traffic",
    "sales",
    "exchange_rate",
    "surface_temperature",
    "stock",
    "electricity"
  ];
  
 
  const handlePredict = async () => {
    setForecastData([]);
    setHistoricalData([]);
    setMetadata({})
    setErrorMetric({});
    setIsLoading(true);
    try {
      // Step 1: Fetch metadata
      const response = await fetch(`https://shabthanaj-backenend-panel.hf.space/forecast/metadata/${selectedDataset}`);
      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }
      const metadataData = await response.json();
      console.log("Metadata received:", metadataData);
      setMetadata(metadataData);

      // Step 2: Fetch historical data
      // const historicalResponse = await fetch(`https://kajaani-fastapi-panel-data.hf.space/historical/data/${selectedDataset}`);
      const historicalResponse = await fetch(`https://shabthanaj-backenend-panel.hf.space/historical/data/${selectedDataset}`);
      if (!historicalResponse.ok) {
        throw new Error(`API Error: ${historicalResponse.status}`);
      }
      const histData = await historicalResponse.json();
      console.log("HistoricalData received:", histData);
      setHistoricalData(histData);

      // Step 3: Trigger forecast generation
      // const postResponse = await fetch(`https://kajaani-fastapi-panel-data.hf.space/forecast/${selectedDataset}`, {
      //   method: "POST",
      //   headers: { "Content-Type": "application/json" }
      // });
      const postResponse = await fetch(`https://shabthanaj-backenend-panel.hf.space/forecast/${selectedDataset}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" }
      });
      if (!postResponse.ok) {
        throw new Error(`Forecast API Error: ${postResponse.status}`);
      }
      const postResult = await postResponse.json();
      console.log("Forecast generation triggered:", postResult);

      // Step 4: Fetch forecast data (GET)
      // const getResponse = await fetch(`https://kajaani-fastapi-panel-data.hf.space/forecast/data/${selectedDataset}`);
      const getResponse = await fetch(`https://shabthanaj-backenend-panel.hf.space/forecast/data/${selectedDataset}`);
      if (!getResponse.ok) {
        throw new Error(`Data API Error: ${getResponse.status}`);
      }
      const forecastData = await getResponse.json();
      console.log("Forecast data received:", forecastData);
      setForecastData(forecastData);

      // Step 5: Fetch error metrics
      // const MetricResponse = await fetch(`https://kajaani-fastapi-panel-data.hf.space/metrics/Enhanced%20TFT/${selectedDataset}`);
      const MetricResponse = await fetch(`https://shabthanaj-backenend-panel.hf.space/metrics/Enhanced%20TFT/${selectedDataset}`);
      if (!MetricResponse.ok) {
        throw new Error(`API Error: ${MetricResponse.status}`);
      }
      const errormetric = await MetricResponse.json();
      console.log("Error metrics received:", errormetric);
      setErrorMetric(errormetric);

    } catch (error) {
      console.error("Prediction error:", error);
    } finally {
      setIsLoading(false);
    }
  };


  return (
    <div className="p-4 container">
      <Header />

      <div className="layout">
        <div className="left-panel">
          <div className="mb-4">
            <label htmlFor="dataset-select" className="block mb-2">Select Dataset:</label>
            <select
              id="dataset-select"
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="p-2 border rounded w-full"
            >
              <option value="">-- Select a dataset --</option>
              {datasets.map(dataset => (
                <option key={dataset} value={dataset}>
                  {dataset.replace(/_/g, ' ')}
                </option>
              ))}
            </select>
          </div>
          {/* <ModelSelector setModel={setModel} /> */}
          <button onClick={handlePredict}>
            Forecast
          </button>
          {selectedDataset && Object.keys(errorMetric).length !== 0 && (
            <p className="metric-box">
              <strong>Mean Absolute Pecentage Error (MAPE):</strong> {errorMetric.MAPE}%
            </p>
          )}

        </div>
        
        {selectedDataset && Object.keys(metadata).length !== 0 && (
          <div className="right-panel">
            <p className="dataset-title">DATASET DETAILS</p>
            
              <>
                <p><strong>Total Entities:</strong> {metadata.num_entities}</p>
                <p><strong>Total Datapoints:</strong> {metadata.total_datapoints}</p>
                <p><strong>Frequency:</strong> {metadata.date_format}</p>
                <p><strong>Panel Type:</strong> {metadata.type}</p>
              </>
          
          </div> 
         )}
      
      </div>

      {isLoading && (
        <div className="loader-container">
          <div className="spinner"></div>
          <p>Forecasting...</p>
        </div>
      )}
      
      
      {!isLoading&&(
        <div className="forecast-chart-container">
          <ForecastChart 
          forecastData={forecastData.data}
          historicalData={historicalData} 
          />
        </div>
      )}
    </div>
  );


}

export default App;
