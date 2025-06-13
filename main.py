from model.data_loader import DataLoader
from model.preprocessing import DataPreprocessor
from model.eda import EDA
from model.features import FeatureEngineer
from model.mlflow_logger import MLflowLogger
from model.file_controller import FileController
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import os
from model.schemas import WeatherRecord
from config import DATA_DIR, DEFAULT_DATA_FILE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = os.path.join(DATA_DIR, DEFAULT_DATA_FILE)

def main():

    # Example: Validate the data file before loading
    if not os.path.exists(DATA_PATH):
        print(f"Data file '{DATA_PATH}' not found. Please add it to the 'data/' folder.")
        return
    # Check file type and size using FileController class
    file_controller = FileController()
    mimetype = 'text/csv'  # You may want to use python-magic or mimetypes for real detection
    size_bytes = os.path.getsize(DATA_PATH)
    warnings = file_controller.validate_upload(mimetype, size_bytes)
    if warnings:
        for w in warnings:
            print(f"Warning: {w}")
        return

    # Data Loading
    loader = DataLoader(DATA_PATH)
    df = loader.load()

    # Validate schema using pydantic
    try:
        for record in df.to_dict(orient='records'):
            WeatherRecord(**record)
    except Exception as e:
        print(f"Schema validation error: {e}")
        return
    # Ensure correct dtypes
    dtype_map = {
        'date': int,
        'cloud_cover': float,
        'sunshine': float,
        'global_radiation': float,
        'max_temp': float,
        'mean_temp': float,
        'min_temp': float,
        'precipitation': float,
        'pressure': float,
        'snow_depth': float
    }
    for col, dtype in dtype_map.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    print("Missing values per column:\n", df.isnull().sum())

    # Data Preprocessing
    preprocessor = DataPreprocessor(df)
    df_imputed = preprocessor.impute_missing()
    print("After imputation, missing values:\n", df_imputed.isnull().sum())

    # EDA
    eda = EDA(df_imputed)
    eda.summary()
    eda.plot_mean_temp_distribution()

    # Additional EDA: correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df_imputed.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # Feature Engineering
    fe = FeatureEngineer(df_imputed)
    features, target = fe.select_features()
    print("Features used for training:")
    print(features.columns)
    X_train, X_test, y_train, y_test = fe.split()
    X_train_scaled, X_test_scaled = fe.scale()
    print("Training set shape:", X_train.shape, y_train.shape)
    print("Test set shape:", X_test.shape, y_test.shape)

    # Modeling & MLflow Logging
    logger = MLflowLogger()
    models = [
        (LinearRegression(), "Linear Regression"),
        (DecisionTreeRegressor(random_state=0), "Decision Tree"),
        (RandomForestRegressor(random_state=0), "Random Forest")
    ]
    for model, name in models:
        logger.train_and_log(model, name, X_train_scaled, y_train, X_test_scaled, y_test)
    experiment_results = logger.show_results()
    # experiment_results is now available for further use

if __name__ == "__main__":
    main()
