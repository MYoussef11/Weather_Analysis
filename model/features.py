import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator

class ImputeLogger(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        nulls = np.isnan(X).sum() if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.number) else pd.isnull(X).sum().sum()
        logging.info(f"After imputation: shape={X.shape}, nulls={nulls}")
        return X

class FeatureEngineer:
    """
    Handles train/test splitting, imputation, and scaling for weather data features and target.
    """
    def __init__(self, features: pd.DataFrame, target: pd.Series):
        """
        Initialize FeatureEngineer with features and target.
        Args:
            features (pd.DataFrame): Feature columns.
            target (pd.Series): Target column.
        """
        self.features = features.copy()
        self.target = target.copy()
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.X_train_imputed = self.X_test_imputed = None
        self.X_train_scaled = self.X_test_scaled = None
        self.scaler = None
        self.imputer = None

        self._validate_features_and_target(context="initialization")

    def _validate_features_and_target(self, context="validation"):
        """
        Checks if features and target are present and non-empty. Logs errors if not.
        Args:
            context (str): Description of where the check is called (for logging).
        Raises:
            ValueError if features/target are None or empty.
        """
        if self.features is None or self.target is None:
            logging.error(f"Features or target are None during {context}.")
            raise ValueError("Features or target are None.")
        if self.features.empty:
            logging.error(f"Features DataFrame is empty during {context}.")
            raise ValueError("Features DataFrame is empty.")
        if self.target.empty:
            logging.error(f"Target Series is empty during {context}.")
            raise ValueError("Target Series is empty.")

    def get_preprocessing_pipeline(self, model_type: str, numeric_features: list):
        """
        Returns a sklearn Pipeline or ColumnTransformer for the given model type (numeric only).
        Args:
            model_type (str): 'linear', 'random_forest', or 'xgboost'
            numeric_features (list): List of numeric feature names
        Returns:
            sklearn Pipeline or ColumnTransformer
        """
        try:
            # Imputation for numeric only
            num_imputer = SimpleImputer(strategy="median")
            # Scaling only for models that need it
            if model_type == "linear":
                preprocessor = ColumnTransformer([
                    ("num", Pipeline([
                        ("imputer", num_imputer),
                        ("impute_logger", ImputeLogger()),
                        ("scaler", StandardScaler())]), numeric_features)
                ])
            elif model_type == "random_forest" or model_type == "xgboost":
                preprocessor = ColumnTransformer([
                    ("num", Pipeline([
                        ("imputer", num_imputer),
                        ("impute_logger", ImputeLogger())]), numeric_features)
                ])
            else:
                logging.error(f"Unknown model_type: {model_type}")
                raise ValueError(f"Unknown model_type: {model_type}")
            logging.info(f"Preprocessing pipeline created for {model_type}.")
            return preprocessor
        except Exception as e:
            logging.error(f"Error creating preprocessing pipeline: {e}")
            raise

    def split(self, train_size=0.7, random_state=0):
        """
        Split features and target into train and test sets.
        Returns:
            X_train, X_test, y_train, y_test
        """
        self._validate_features_and_target(context="split")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, train_size=train_size, random_state=random_state, shuffle=False
        )
        logging.info(f"Data split: X_train shape {self.X_train.shape}, X_test shape {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test
