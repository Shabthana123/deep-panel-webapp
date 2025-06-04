# 🌌 Panel Time Series Forecasting App

A web application that showcases Panel Time Series Forecasting using an advanced Transformer model—an enhanced version of the Temporal Fusion Transformer (TFT). This model integrates multi-scale series decomposition, segment-wise attention, cross-entity attention, and an adaptive weighting mechanism for improved forecasting performance.
Select from **11 diverse datasets**, forecast future values, and visualize results for each entity with **7 quantiles** forecasted **30 timesteps ahead**, displayed in animated and interactive plots.  
Built with **React (Vite)** and powered by **Hugging Face APIs** for model inference.

---

## 🚀 Features

- **Select Datasets**: Choose from 11 preloaded panel time series datasets across domains like economics, healthcare, and finance.
- **One-Click Forecast**: Hit the **Forecast** button to trigger deep learning predictions for all entities in the dataset.
- **Quantile Forecasts**: Visualize **7 quantile predictions** per entity, plotted separately, forecasting **30 timesteps into the future**.
- **Interactive Visualizations**: Animations and clear charts help explore model outputs dynamically.
- **Dataset Details**: View metadata and summary information for the selected dataset.

---

## 🛠️ Tech Stack

- **Frontend**: Vite + React + Plotly.js
- **Backend API**: Hugging Face Inference API
- **Styling**: CSS

---

## 📊 How It Works

1. **Select a Dataset** from the dropdown.
2. Click **Forecast** to generate predictions.
3. View:
   - Dataset details.(Total Entities,Total Datapoints,Frequency,Panel Type)
   - Forecast plots for each entity.
   - Animated quantile forecasts over time.

---

## 📸 Screenshots

![image](https://github.com/user-attachments/assets/61a4e7e5-a7c4-4685-934e-ad4d7535f2ee)
![image](https://github.com/user-attachments/assets/9f4a1085-f9b2-489a-ba43-30d345b39dcf)

---

## 📦 Setup Instructions

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/Shabthana123/deep-panel-webapp.git
    cd deep-panel-webapp/frontend
    ```

2. **Install Dependencies**:

    ```bash
    npm install
    ```

3. **Start the App**:

    ```bash
    npm run dev
    ```

---

## 🙌 Acknowledgments

- Hugging Face for model APIs
- Plotly.js for visualization tools
- Vite + React for the frontend framework

---

