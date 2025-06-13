import mlflow
import mlflow.sklearn
from sklearn.metrics import root_mean_squared_error

class MLflowLogger:
    def __init__(self):
        mlflow.sklearn.autolog()

    def train_and_log(self, model, model_name, X_train, y_train, X_test, y_test):
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = root_mean_squared_error(y_test, y_pred)
            mlflow.log_metric("rmse", rmse)
            print(f"{model_name} RMSE: {rmse:.4f}")
        return rmse

    def show_results(self):
        experiment_results = mlflow.search_runs()
        print("Experiment Results:")
        print(experiment_results[["run_id", "metrics.rmse"]])
        return experiment_results
