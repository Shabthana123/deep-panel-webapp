// import React, { useState } from "react";
// import Header from "./components/Header";
// import DatasetSelector from "./components/DatasetSelector";
// import ModelSelector from "./components/ModelSelector";
// import ForecastChart from "./components/ForecastChart";
// import MetricsDisplay from "./components/MetricsDisplay";
// import axios from "axios";
// import './App.css' 

// function App() {
//   const [file, setFile] = useState(null);
//   const [model, setModel] = useState("TFT");
//   const [results, setResults] = useState(null);

//   const handlePredict = async () => {
//     const formData = new FormData();
//     formData.append("file", file);
//     formData.append("model_name", model);

//     const res = await axios.post("http://localhost:8000/predict/", formData);
//     setResults(res.data);
//   };

//   return (
//     <div className="p-4">
//       <Header />
//       <DatasetSelector setFile={setFile} />
//       <ModelSelector setModel={setModel} />
//       <button onClick={handlePredict} className="mt-4 p-2 bg-blue-500 text-white">
//         Predict
//       </button>
//       {results && (
//         <>
//           <MetricsDisplay mape={results.mape} />
//           <ForecastChart actual={results.actual} predicted={results.predicted} />
//         </>
//       )}
//     </div>
//   );
// }

// export default App;


import React, { useState } from "react";
import Header from "./components/Header";
import DatasetSelector from "./components/DatasetSelector";
import ModelSelector from "./components/ModelSelector";
import ForecastChart from "./components/ForecastChart";
import MetricsDisplay from "./components/MetricsDisplay";
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [model, setModel] = useState("TFT");

  // ✅ Manual dummy results
  const [results, setResults] = useState({
  actual: [100, 120, 130, 125, 140, 150, 160, 170, 175, 180],
  predicted: [98, 118, 134, 122, 138, 148, 158, 169, 176, 182],
  mape: 2.63,
  quantile: {
    10: [90, 105, 115, 110, 130, 135, 145, 155, 160, 170],
    50: [100, 120, 130, 125, 140, 150, 160, 170, 175, 180],
    90: [110, 135, 145, 140, 150, 160, 170, 185, 190, 195]
  }
});


  const handlePredict = async () => {
    // Simulated API delay
    setTimeout(() => {
      setResults({
        actual: [100, 120, 130, 125, 140, 150, 160, 170, 175, 180],
        predicted: [98, 118, 134, 122, 138, 148, 158, 169, 176, 182],
        mape: 2.63,
        quantile: [105, 125, 135, 130, 145, 153, 165, 174, 178, 185]
      });
    }, 500);
  };

  return (
    <div className="p-4 container">
      <Header />
      <DatasetSelector setFile={setFile} />
      <ModelSelector setModel={setModel} />
      <button onClick={handlePredict} className="mt-4 p-2 bg-blue-500 text-white">
        Predict
      </button>
      {results && (
        <>
          <MetricsDisplay mape={results.mape} />
          <ForecastChart
            actual={results.actual}
            predicted={results.predicted}
            quantiles={results.quantile}
          />
        </>
      )}
    </div>
  );
}

export default App;
