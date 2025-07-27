import mlflow
import mlflow.sklearn
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
import logging


class MLflowLogger:
    """
    Handles MLflow experiment setup, model training, logging, and result display for weather regression.
    """
    def __init__(self):
        """
        Initialize MLflowLogger and set up a unique experiment name.
        """
        try:
            mlflow.sklearn.autolog()
            experiment_name = f"WeatherRegression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logging.error(f"Error initializing MLflowLogger: {e}")

    def log_cv_results(self, model_name, search, today):
        """
        Log best params, best score, and best model from a CV search to MLflow.
        Args:
            model_name (str): Name of the model/estimator.
            search: Fitted GridSearchCV or RandomizedSearchCV object.
            today (str): Date string for tagging.
        """
        try:
            with mlflow.start_run(run_name=model_name):
                mlflow.set_tag("date", today)
                mlflow.log_params(search.best_params_)
                mlflow.log_metric("best_rmse", -search.best_score_)
                mlflow.sklearn.log_model(search.best_estimator_, f"{model_name}_best")
                logging.info(f"{model_name} best RMSE: {-search.best_score_:.4f}")
        except Exception as e:
            logging.error(f"Error logging CV results for {model_name}: {e}")
    def train_and_log(self, model, model_type, X_train, y_train, X_test, y_test, best_params=None):
        """
        Train a model, log RMSE, R² score, and MAE to MLflow, and log the results.
        Args:
            model: scikit-learn pipeline or model instance (numeric features only).
            model_type (str): Model type for the run (e.g., 'linear', 'random_forest', 'xgboost').
            X_train, y_train: Training data.
            X_test, y_test: Test data.
            best_params: dict, best hyperparameters from GridSearchCV (optional)
        Returns:
            dict: Metrics (train_rmse, train_r2, train_mae, rmse, r2, mae, best_params)
        """
        try:
            with mlflow.start_run(run_name=model_type):
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                train_rmse = root_mean_squared_error(y_train, y_train_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                rmse = root_mean_squared_error(y_test, y_test_pred)
                r2 = r2_score(y_test, y_test_pred)
                mae = mean_absolute_error(y_test, y_test_pred)
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("target_column", "mean_temp")
                if hasattr(model, "steps"):
                    mlflow.log_param("pipeline_steps", [name for name, _ in model.steps])
                if best_params:
                    mlflow.log_params(best_params)
                mlflow.log_metric("train_rmse", train_rmse)
                mlflow.log_metric("train_r2", train_r2)
                mlflow.log_metric("train_mae", train_mae)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("mae", mae)
                mlflow.sklearn.log_model(model, f"{model_type}_pipeline")
                metrics_dict = {
                    "mean_absolute_error_X_train": train_mae,
                    "r2_score_X_train": train_r2,
                    "root_mean_squared_error_X_train": train_rmse,
                    "mean_absolute_error_X_test": mae,
                    "r2_score_X_test": r2,
                    "root_mean_squared_error_X_test": rmse
                }
                try:
                    mlflow.log_dict(metrics_dict, f"metric_info_{model_type}.json")
                    logging.info(f"Metrics JSON logged to MLflow for {model_type}")
                except Exception as e:
                    logging.error(f"Error saving/logging metrics JSON artifact for {model_type}: {e}")
                logging.info(f"{model_type} Train RMSE: {train_rmse:.4f}, Train R²: {train_r2:.4f}, Train MAE: {train_mae:.4f}")
                logging.info(f"{model_type} Eval RMSE: {rmse:.4f}, Eval R²: {r2:.4f}, Eval MAE: {mae:.4f}")
            return {"train_rmse": train_rmse, "train_r2": train_r2, "train_mae": train_mae, "rmse": rmse, "r2": r2, "mae": mae, "best_params": best_params}
        except Exception as e:
            logging.error(f"Error in train_and_log for {model_type}: {e}")
            return None

    def show_results(self):
        """
        Display and return MLflow experiment results (run_id and RMSE).
        Returns:
            DataFrame: MLflow experiment results.
        """
        try:
            experiment_results = mlflow.search_runs()
            logging.info("Experiment Results:")
            logging.info(f"\n{experiment_results[['run_id', 'metrics.rmse']]}")
            return experiment_results
        except Exception as e:
            logging.error(f"Error showing MLflow results: {e}")
            return None
