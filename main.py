import os
import logging
import mimetypes
from datetime import datetime
from config import DATA_DIR, DEFAULT_DATA_FILE
from model.data_loader import DataLoader
from model.preprocessing import DataPreprocessor
from model.eda import EDA
from model.features import FeatureEngineer
from model.mlflow_logger import MLflowLogger
from model.file_controller import FileController
from model.schemas import WeatherRecord
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

DATA_PATH = os.path.join(DATA_DIR, DEFAULT_DATA_FILE)

def main():

    # Example: Validate the data file before loading
    if not os.path.exists(DATA_PATH):
        logging.error(f"Data file '{DATA_PATH}' not found. Please add it to the 'data/' folder.")
        return
    # Check file type and size using FileController class
    file_controller = FileController()
    mimetype, _ = mimetypes.guess_type(DATA_PATH)
    if mimetype is None:
        mimetype = 'application/octet-stream'
    # Accept both 'text/csv' and 'application/vnd.ms-excel' as valid CSV mimetypes
    allowed_types = ['text/csv', 'application/vnd.ms-excel']
    if mimetype not in allowed_types:
        logging.warning(f"Warning: File type '{mimetype}' is not a standard CSV type. Allowed types: {allowed_types}. Proceeding anyway.")
    size_bytes = os.path.getsize(DATA_PATH)
    warnings = file_controller.validate_upload(mimetype, size_bytes)
    if warnings:
        for w in warnings:
            logging.warning(f"Warning: {w}")
        # Do not return; continue execution even if there are warnings

    # Data Loading
    loader = DataLoader(DATA_PATH)
    df = loader.load()

    # Validate schema using pydantic
    try:
        for record in df.to_dict(orient='records'):
            WeatherRecord(**record)
    except Exception as e:
        logging.error(f"Schema validation error: {e}")
        return
    # Ensure correct dtypes using WeatherRecord schema (Pydantic v2+)
    dtype_map = {name: field.annotation for name, field in WeatherRecord.model_fields.items()}
    for col, dtype in dtype_map.items():
        if col in df.columns:
            # Convert pydantic types to Python types
            if dtype is int:
                df[col] = df[col].astype(int)
            elif dtype is float:
                df[col] = df[col].astype(float)
            elif dtype is str:
                df[col] = df[col].astype(str)

    # EDA on raw data before preprocessing
    eda_raw = EDA(df)
    eda_raw.summary()
    eda_raw.missing_data_summary()
    eda_raw.plot_mean_temp_distribution()
    eda_raw.boxplots()
    eda_raw.plot_correlation_heatmap()

    # Feature selection
    preprocessor = DataPreprocessor(df)
    features, target = preprocessor.select_features()
    logging.info("Features used for training:")
    logging.info(f"{features.columns}")

    # Automatically detect numeric features only
    numeric_features = features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    logging.info(f"Numeric features: {numeric_features}")

    # Feature Engineering (split)
    fe = FeatureEngineer(features, target)
    X_train, X_test, y_train, y_test = fe.split()

    # Modeling & MLflow Logging
    logger = MLflowLogger()
    models = [
        (LinearRegression(), "Linear Regression", "linear"),
        (SVR(), "Support Vector Regression", "linear"),
        (DecisionTreeRegressor(random_state=0), "Decision Tree", "random_forest"),
        (RandomForestRegressor(random_state=0), "Random Forest", "random_forest"),
        (XGBRegressor(random_state=0, n_jobs=-1), "XGBoost", "xgboost")
    ]

    param_grids = {
        "Linear Regression": {},
        "Support Vector Regression": {"regressor__C": [0.1, 1, 10], "regressor__kernel": ["linear", "rbf"]},
        "Decision Tree": {"regressor__max_depth": [3, 5, 10]},
        "Random Forest": {"regressor__n_estimators": [50, 100], "regressor__max_depth": [3, 5, 10]},
        "XGBoost": {"regressor__n_estimators": [50, 100], "regressor__max_depth": [3, 5, 10]}
    }
    def fmt(val):
        try:
            return f"{float(val):.4f}"
        except (ValueError, TypeError):
            return str(val)
    backup_path = os.path.join("artifacts", "model_results_backup.txt")
    for model, model_name, model_type in models:
        preprocessor = fe.get_preprocessing_pipeline(model_type, numeric_features)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])
        param_grid = param_grids.get(model_name, {})
        if param_grid:
            search = GridSearchCV(pipeline, param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
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
    experiment_results = logger.show_results()
    # experiment_results is now available for further use

if __name__ == "__main__":
    main()
