import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from datetime import datetime

"""
Deployment Script

Loads today's weather features, predicts mean temperature using the trained Random Forest pipeline from MLflow, and (optionally) evaluates prediction if the true value is available.
"""

# Configuration
MODEL_RUN_ID = "f034353edda14a79a0284b931d31979a"  # Update with your actual run ID
PIPELINE_NAME = "Random Forest_pipeline"  # Update if needed
MODEL_URI = f"runs:/{MODEL_RUN_ID}/{PIPELINE_NAME}"
ARTIFACT_PATH = "artifacts/deploy_results_backup.txt"

# Example input: today's weather features
# Replace these values with real-time or user input as needed
TODAY_INPUT = {
    "cloud_cover": 1,             # oktas
    "sunshine": 4.5,              # hours
    "global_radiation": 550,      # W/m2
    "max_temp": 22,               # °C
    "min_temp": 14,               # °C
    "precipitation": 1.2,         # mm
    "pressure": 101900,           # Pa
    "snow_depth": 0               # cm
}


def predict_mean_temp(input_dict, model_uri):
    """Predict mean temperature from input features using a trained MLflow pipeline."""
    X = pd.DataFrame([input_dict])
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from MLflow: {e}")
    pred = model.predict(X)[0]
    return pred


def evaluate_prediction(y_true, y_pred):
    """Compute evaluation metrics for a single prediction."""
    mae = mean_absolute_error([y_true], [y_pred])
    rmse = root_mean_squared_error([y_true], [y_pred])
    # R² is only defined for more than one sample
    if len([y_true]) > 1:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = None
    return mae, rmse, r2

def save_results(results, path):
    """Save deploy evaluation results to artifact file."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(path, "a", encoding="utf-8") as f:
        for entry in results:
            model_name, rmse, mae, r2 = entry
            r2_str = f"{r2:.4f}" if r2 is not None and not np.isnan(r2) else "Not defined for single sample"
            f.write(
                f"\nDate: {now}\nModel: {model_name}"
                f"\nDeploy RMSE: {rmse:.4f}"
                f"\nDeploy MAE: {mae:.4f}"
                f"\nDeploy R²: {r2_str}"
                f"\n{'-'*40}\n"
            )


def main():
    """Main entry point for deployment prediction and evaluation."""
    print("Loading model and predicting today's mean temperature...")
    try:
        predicted_mean_temp = predict_mean_temp(TODAY_INPUT, MODEL_URI)
        print(f" Predicted mean temperature: {predicted_mean_temp:.2f}°C")
    except Exception as e:
        print(f"Prediction failed: {e}")
        return
    # If you have the true observed mean temperature for today, evaluate
    actual_mean_temp = 18  # Replace with the true value if available
    if actual_mean_temp is not None:
        print("\nEvaluation on Today's Data:")
        results = []
        model_name= "Random Forest"  # Update if needed
        try:
            mae, rmse, r2 = evaluate_prediction(actual_mean_temp, predicted_mean_temp)
            print(f"  MAE : {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            if r2 is not None and not np.isnan(r2):
                print(f"  R²  : {r2:.4f}")
            else:
                print("  R²  : Not defined for single sample.")
            results.append((model_name, rmse, mae, r2))
        except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
        save_results(results, ARTIFACT_PATH)
        print(f"\nResults saved to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
