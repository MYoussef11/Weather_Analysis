import os
import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

"""
Holdout Validation Script

Evaluates selected trained models (XGBoost, LinearRegression, RandomForest) on an unseen holdout dataset and saves results to artifacts/holdout_results_backup.txt.
"""

HOLDOUT_CSV = "data/london_weather_data_2021_to_2023.csv"
ARTIFACT_PATH = "artifacts/holdout_results_backup.txt"

# Models to evaluate: (run_id, pipeline_name, display_name)
MODELS = [
    ("5a6bc60497b64ee1bec37f1a86ab7c4b", "XGBoost_pipeline", "XGBoost"),
    ("14e7058a1efc4f0f9d22c9f2213c5f7b", "Linear Regression_pipeline", "Linear Regression"),
    ("f034353edda14a79a0284b931d31979a", "Random Forest_pipeline", "Random Forest")
]

def load_holdout_data(path):
    """Load and preprocess holdout dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Holdout file not found: {path}")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    if "mean_temp" not in df.columns:
        raise KeyError("Expected target column 'mean_temp' not found in holdout data")
    X = df.drop(columns=["mean_temp"])
    y = df["mean_temp"]
    return X, y

def evaluate_model(run_id, pipeline_name, X, y):
    """Load model from MLflow and evaluate on holdout data."""
    model_uri = f"runs:/{run_id}/{pipeline_name}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{pipeline_name}' (run_id={run_id}): {e}")
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return rmse, mae, r2

def save_results(results, path):
    """Save holdout evaluation results to artifact file."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(path, "a", encoding="utf-8") as f:
        for entry in results:
            model_name, rmse, mae, r2 = entry
            f.write(
                f"\nDate: {now}\nModel: {model_name}"
                f"\nHoldout RMSE: {rmse:.4f}"
                f"\nHoldout MAE: {mae:.4f}"
                f"\nHoldout R²: {r2:.4f}"
                f"\n{'-'*40}\n"
            )

def main():
    """Main entry point for holdout validation."""
    try:
        X_holdout, y_holdout = load_holdout_data(HOLDOUT_CSV)
        print(f"Holdout data loaded. Features: {X_holdout.shape}, Target: {y_holdout.shape}")
        results = []
        for run_id, pipeline_name, model_name in MODELS:
            print(f"\nEvaluating {model_name}...")
            try:
                rmse, mae, r2 = evaluate_model(run_id, pipeline_name, X_holdout, y_holdout)
                print(f"  RMSE: {rmse:.4f}")
                print(f"   MAE: {mae:.4f}")
                print(f"    R²: {r2:.4f}")
                results.append((model_name, rmse, mae, r2))
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
        save_results(results, ARTIFACT_PATH)
        print(f"\nResults saved to {ARTIFACT_PATH}")
    except Exception as e:
        print(f"Holdout validation failed: {e}")

if __name__ == "__main__":
    main()
