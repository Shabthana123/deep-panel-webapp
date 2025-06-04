# app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
import json
import pandas as pd

# Add parent path for local imports
import sys
sys.path.append(str(Path(__file__).parent.resolve()))

# Import functions from your script
from app.model.forecasting_script import (
    DATASET_CONFIGS,
    forecast_next_30_steps,
    load_and_preprocess_data
)

# Initialize FastAPI app
app = FastAPI(
    title="Time Series Forecast API",
    description="API for forecasting time series using Temporal Fusion Transformer"
)

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths
BASE_DIR = Path(__file__).resolve().parent  # Now 'app' folder
FORECAST_DIR = BASE_DIR / "forecasts"
METRICS_FILE = BASE_DIR.parent / "data" / "Error_Metrics.csv"

@app.get("/")
def read_root():
    return {"message": "Welcome to Time Series Forecast API"}

@app.get("/datasets")
def get_datasets():
    """List all available datasets"""
    return {"datasets": list(DATASET_CONFIGS.keys())}

@app.post("/forecast/{dataset_key}")
def run_forecast(dataset_key: str):
    """Trigger forecasting for a specific dataset"""
    if dataset_key not in DATASET_CONFIGS:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        print(f"Running forecast for {dataset_key}...")
        forecasts = forecast_next_30_steps(dataset_key)

        if forecasts is None:
            raise HTTPException(status_code=500, detail="Forecast failed")

        # Save forecasts to JSON
        output_json = FORECAST_DIR / f"{dataset_key}_forecasts.json"
        forecasts.to_json(output_json, orient="records", date_format="iso")

        # Find plot image
        config = DATASET_CONFIGS[dataset_key]
        plot_name = f"{config['dataset_name'].lower().replace(' ', '_')}_forecasts_plot.png"
        plot_path = FORECAST_DIR / plot_name

        return {
            "status": "success",
            "forecast_file": str(output_json),
            "plot_file": str(plot_path) if os.path.exists(plot_path) else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecast/data/{dataset_key}")
def get_forecast_data(dataset_key: str):
    """Get forecasted data as JSON"""
    forecast_file = FORECAST_DIR / f"{dataset_key}_forecasts.json"
    if not forecast_file.exists():
        raise HTTPException(status_code=404, detail="Forecast data not found")

    with open(forecast_file, "r") as f:
        data = json.load(f)

    return {"data": data}

@app.get("/forecast/plot/{dataset_key}")
def get_forecast_plot(dataset_key: str):
    """Return the forecast plot image"""
    config = DATASET_CONFIGS[dataset_key]
    plot_name = f"{config['dataset_name'].lower().replace(' ', '_')}_forecasts_plot.png"
    plot_path = FORECAST_DIR / plot_name

    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Plot not found")

    return FileResponse(plot_path, media_type="image/png", filename=plot_name)

@app.get("/forecast/metadata/{dataset_key}")
def get_forecast_metadata(dataset_key: str):
    """Return metadata about the dataset including size, entities, and column info"""
    if dataset_key not in DATASET_CONFIGS:
        raise HTTPException(status_code=404, detail="Dataset not found")

    config = DATASET_CONFIGS[dataset_key]
    data_path = BASE_DIR.parent / "data" / config["data_file"]

    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"Data file not found at {data_path}")

    try:
        # Load CSV
        df = pd.read_csv(data_path)

        # Rename columns based on config
        col_config = config['columns']
        df.columns = col_config  # Direct assignment from config

        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Handle special cases (e.g., sales)
        if dataset_key == 'sales':
            df['Entity'] = df['Entity'].astype(str)

        # Compute metadata
        total_datapoints = len(df)
        entities = df['Entity'].unique().tolist()
        num_entities = len(entities)

        datapoints_per_entity = {}
        for entity in entities:
            count = len(df[df['Entity'] == entity])
            datapoints_per_entity[entity] = count

        time_idx_type = config['time_idx_type']  # daily/monthly/annual
        columns_used = col_config

        return {
            "dataset_key": dataset_key,
            "dataset_name": config["dataset_name"],
            "total_datapoints": total_datapoints,
            "num_entities": num_entities,
            "entities": entities,
            "datapoints_per_entity": datapoints_per_entity,
            "date_format": time_idx_type,
            "columns_used": columns_used,
            "extra_columns": list(df.columns[3:]),  # Show extra columns beyond Entity/Date/Value
            'type': config['type']
            
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/metrics/{model_file}/{dataset_key}")
def get_model_metrics(model_file: str, dataset_key: str):
    """
    Get error metrics for a specific model and dataset
    Example: /metrics/co2.ckpt/co2
    """
    if dataset_key not in DATASET_CONFIGS:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Check if metrics file exists
    if not METRICS_FILE.exists():
        raise HTTPException(status_code=404, detail=f"Metrics file not found at {METRICS_FILE}")

    try:
        # Load metrics data
        df = pd.read_csv(METRICS_FILE)

        # Filter by model and dataset
        result = df[(df['Model'] == model_file) & (df['Dataset'] == dataset_key)]

        if result.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No metrics found for Model={model_file}, Dataset={dataset_key}"
            )

        # Return the first matching row as JSON
        return result.iloc[0].to_dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/historical/data/{dataset_key}")
def get_historical_data(dataset_key: str):
    if dataset_key not in DATASET_CONFIGS:
        raise HTTPException(status_code=404, detail="Dataset not found")

    config = DATASET_CONFIGS[dataset_key]
    data_path = BASE_DIR.parent / "data" / config["data_file"]

    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"Data file not found at {data_path}")

    try:
        # Load and preprocess data
        df, _ = load_and_preprocess_data(dataset_key)
        
        result = []

        for entity in df['Entity'].unique():
            entity_df = df[df['Entity'] == entity].sort_values('Date')
            # Limit to 200 most recent records if more than 200
            limited_df = entity_df.tail(200) if len(entity_df) > 200 else entity_df

            result.append({
                "entity": entity,
                "values": limited_df[['Date', 'Value']].to_dict(orient='records')
            })

        return {"status": "success", "data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/historical/data/{dataset_key}")
# def get_historical_data(dataset_key: str):
#     if dataset_key not in DATASET_CONFIGS:
#         raise HTTPException(status_code=404, detail="Dataset not found")

#     config = DATASET_CONFIGS[dataset_key]
#     data_path = BASE_DIR.parent / "data" / config["data_file"]

#     if not data_path.exists():
#         raise HTTPException(status_code=404, detail=f"Data file not found at {data_path}")

#     try:
#         # Load and preprocess data
#         df, _ = load_and_preprocess_data(dataset_key)
        
#         result = []

#         for entity in df['Entity'].unique():
#             entity_df = df[df['Entity'] == entity].sort_values('Date').tail(30)

#             result.append({
#                 "entity": entity,
#                 "values": entity_df[['Date', 'Value']].to_dict(orient='records')
#             })

#         return {"status": "success", "data": result}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
# myenv\Scripts\activate
