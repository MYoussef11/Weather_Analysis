import os
import logging
import joblib
import mimetypes
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from model.config import DATA_DIR, DEFAULT_DATA_FILE
from model.data_loader import DataLoader
from model.preprocessing import DataPreprocessor
from model.features import FeatureEngineer
from model.mlflow_logger import MLflowLogger
from model.schemas import WeatherRecord
from model.file_controller import FileController
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="This Pipeline instance is not fitted yet.*"
)


def fmt(val):
    try:
        return f"{float(val):.4f}"
    except (ValueError, TypeError):
        return str(val)


def save_model(model, model_name):
    """Save the trained model to artifacts directory."""
    model_path = os.path.join("artifacts", f"{model_name.replace(' ', '_').lower()}_model.joblib")
    try:
        joblib.dump(model, model_path)
        logging.info(f"Saved {model_name} model to {model_path}")
    except Exception as e:
        logging.error(f"Error saving model {model_name}: {e}")


def log_feature_importance(model, model_name, feature_names):
    """Log feature importance for the trained model."""
    output_path = os.path.join("artifacts", f"{model_name.replace(' ', '_').lower()}_feature_importance.txt")
    try:
        # Get regressor step if model is inside pipeline
        if hasattr(model, "named_steps"):
            model = model.named_steps["regressor"]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            importance_dict = dict(zip(feature_names, importances))
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

            with open(output_path, "w", encoding="utf-8") as f:
                for name, val in sorted_importance:
                    f.write(f"{name}: {val:.4f}\n")
            logging.info(f"Feature importance for {model_name} saved to {output_path}")
        else:
            logging.warning(f"{model_name} does not support feature_importances_")
    except Exception as e:
        logging.error(f"Error logging feature importance for {model_name}: {e}")


def main():
    """Main entry point for training models."""

    DATA_PATH = os.path.join(DATA_DIR, DEFAULT_DATA_FILE)
    if not os.path.exists(DATA_PATH):
        logging.error(f"Data file '{DATA_PATH}' not found. Please add it to the 'data/' folder.")
        return

    # Validate file type and size
    file_controller = FileController()
    mimetype, _ = mimetypes.guess_type(DATA_PATH)
    if mimetype is None:
        mimetype = 'application/octet-stream'
    allowed_types = ['text/csv', 'application/vnd.ms-excel']
    if mimetype not in allowed_types:
        logging.warning(f"File type '{mimetype}' is not standard CSV. Proceeding anyway.")

    size_bytes = os.path.getsize(DATA_PATH)
    warnings = file_controller.validate_upload(mimetype, size_bytes)
    if warnings:
        for w in warnings:
            logging.warning(f"Warning: {w}")

    loader = DataLoader(DATA_PATH)
    df = loader.load()

    try:
        for record in df.to_dict(orient='records'):
            WeatherRecord(**record)
    except Exception as e:
        logging.error(f"Schema validation error: {e}")
        return

    dtype_map = {name: field.annotation for name, field in WeatherRecord.model_fields.items()}
    for col, dtype in dtype_map.items():
        if col in df.columns:
            if dtype is int:
                df[col] = df[col].astype(int)
            elif dtype is float:
                df[col] = df[col].astype(float)
            elif dtype is str:
                df[col] = df[col].astype(str)

    # Initialize preprocessor and feature engineer
    preprocessor = DataPreprocessor(df)
    features, target = preprocessor.select_features()
    numeric_features = features.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Split data into training and testing sets
    fe = FeatureEngineer(features, target)
    X_train, X_test, y_train, y_test = fe.split()

    # Initialize MLflow logger
    logger = MLflowLogger()
    models = [
        (RandomForestRegressor(random_state=0), "Random Forest", "random_forest"),
        (XGBRegressor(random_state=0, n_jobs=-1), "XGBoost", "xgboost")
    ]

    # Define hyperparameter grids for model tuning
    param_grids = {
        "Random Forest": {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [None, 10, 20],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4],
            'regressor__max_features': ['sqrt', 'log2', None],
            'regressor__bootstrap': [True, False]
        },
        "XGBoost": {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__max_depth': [3, 5, 7, 9],
            'regressor__subsample': [0.7, 0.8, 1.0],
            'regressor__colsample_bytree': [0.7, 0.9, 1.0],
            'regressor__reg_alpha': [0, 0.1, 1],
            'regressor__reg_lambda': [0.5, 1, 2]
        }
    }

    # Create artifacts directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)
    backup_path = os.path.join("artifacts", "model_results_v2.txt")

    # Train and evaluate models
    for model, model_name, model_type in models:
        preprocessor = fe.get_preprocessing_pipeline(model_type, numeric_features)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])

        # Perform hyperparameter tuning if applicable
        param_grid = param_grids.get(model_name, {})
        if param_grid:
            search = RandomizedSearchCV(pipeline, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1, error_score="raise")
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            logging.info(f"Best params for {model_name}: {best_params}")
        else:
            best_model = pipeline.fit(X_train, y_train)
            best_params = None

        # Calculate metrics and log
        metrics = logger.train_and_log(best_model, model_name, X_train, y_train, X_test, y_test, best_params=best_params)
        if metrics:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            backup_entry = (
                f"\nDate: {now}\nModel: {model_name}"
                f"\nBest Params: {metrics.get('best_params', 'N/A')}"
                f"\nTrain RMSE: {fmt(metrics.get('train_rmse','N/A'))}"
                f"\nTrain R²: {fmt(metrics.get('train_r2','N/A'))}"
                f"\nTrain MAE: {fmt(metrics.get('train_mae','N/A'))}"
                f"\nEval RMSE: {fmt(metrics.get('rmse','N/A'))}"
                f"\nEval R²: {fmt(metrics.get('r2','N/A'))}"
                f"\nEval MAE: {fmt(metrics.get('mae','N/A'))}"
                f"\n{'-'*40}\n"
            )
            with open(backup_path, "a", encoding="utf-8") as f:
                f.write(backup_entry)

        # Save model and feature importance
        save_model(best_model, model_name)
        log_feature_importance(best_model, model_name, X_train.columns)

    logger.show_results()

if __name__ == "__main__":
    main()
