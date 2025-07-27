import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from datetime import datetime
import os

"""
Deployment Script - Batch Evaluation for Last 10 Days

Loads weather data for the last 10 days, predicts mean temperatures using a trained MLflow model,
and evaluates performance using standard regression metrics.
"""

# === Configuration ===
MODEL_RUN_ID = "5a6bc60497b64ee1bec37f1a86ab7c4b"  # Update with your actual run ID
PIPELINE_NAME = "XGBoost_pipeline" # Update if needed
MODEL_URI = f"runs:/{MODEL_RUN_ID}/{PIPELINE_NAME}"
DEPLOY_CSV = "data/Last 10 days London weather.csv"
ARTIFACT_PATH = "artifacts/deploy_results_batch.txt"
MODEL_NAME = "XGBoost"  # Update if needed
mlflow.set_tracking_uri("http://localhost:5000")

# Load deploy data
def load_deploy_data(path):
    """Load and preprocess deploy dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Deploy file not found: {path}")

    df = pd.read_csv(path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    if "mean_temp" not in df.columns:
        raise KeyError("Expected target column 'mean_temp' not found in data")

    X = df.drop(columns=["mean_temp"])
    y = df["mean_temp"]
    return X, y


# Predict
def predict_mean_temp(X, model_uri):
    """Predict mean temperatures using a trained MLflow model."""
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from MLflow: {e}")

    y_pred = model.predict(X)
    return y_pred


# Evaluate
def evaluate_predictions(y_true, y_pred):
    """Compute evaluation metrics for batch prediction."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


# Save Results
def save_results(model_name, mae, rmse, r2, path):
    """Save deploy evaluation results to artifact file."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    r2_str = f"{r2:.4f}" if not np.isnan(r2) else "Not defined"

    with open(path, "a", encoding="utf-8") as f:
        f.write(
            f"\nDate: {now}\nModel: {model_name}"
            f"\nDeploy RMSE: {rmse:.4f}"
            f"\nDeploy MAE: {mae:.4f}"
            f"\nDeploy R²: {r2_str}"
            f"\n{'-'*40}\n"
        )


def main():
    print("Loading data and model for 10-day batch evaluation...")
    try:
        X, y_true = load_deploy_data(DEPLOY_CSV)
        y_pred = predict_mean_temp(X, MODEL_URI)
    except Exception as e:
        print(f"Failed during loading or prediction: {e}")
        return

    print("Evaluating predictions...")
    try:
        mae, rmse, r2 = evaluate_predictions(y_true, y_pred)
        print(f"\nEvaluation Results (Last 10 Days):")
        print(f"  MAE : {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²  : {r2:.4f}")
        save_results(MODEL_NAME, mae, rmse, r2, ARTIFACT_PATH)
        print(f"\nResults saved to {ARTIFACT_PATH}")
    except Exception as e:
        print(f"Error during evaluation: {e}")


if __name__ == "__main__":
    main()
