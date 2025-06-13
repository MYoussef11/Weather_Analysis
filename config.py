import os

# General project configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')

# Data file name (default)
DEFAULT_DATA_FILE = 'london_weather.csv'

# MLflow
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'WeatherRegression')

# Database (PostgreSQL example)
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'weather_regression')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'your_username')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'your_password')
