# Weather Regression Project

This project demonstrates a complete end-to-end machine learning workflow for weather data regression using Python. It includes modules for data validation, preprocessing, model training, evaluation, and experiment tracking with MLflow. The project is designed to be modular, reusable, and production-ready.

## Features
- Data loading and validation, including file size and format checks
- Schema validation using Pydantic
- Feature selection and preprocessing pipelines
- Modular and object-oriented design
- Support for multiple regression models:
  - Version 1: Linear Regression, Decision Tree, SVR, Random Forest, XGBoost
  - Version 2: Random Forest and XGBoost only, with hyperparameter tuning
- Hyperparameter tuning using `RandomizedSearchCV`
- MLflow integration for experiment tracking (metrics, parameters, models, and artifacts)
- Saving models with `joblib`
- Exporting feature importance to text files
- Holdout validation on unseen test data
- Backup of metrics and results to local artifact folder

## Setup
1. **Create a virtual environment:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Run MLflow UI (optional):**
   ```powershell
   mlflow ui
   ```
4. **Run the main workflow:**
   1.**Train Version 2 models (Random Forest and XGBoost):**
      ```powershell
      python model/model_v2/train_v2.py
      ```

   2.**Train Version 1 models (Linear Regression, SVR, Decision Tree, etc.):**
      ```powershell
      python model/model_v1/main.py
      ```

   3.**Evaluate models on holdout data:**
      ```powershell
      python -m model.holdout_validation.holdout_validation
      ```

## Project Structure
- `model/` — OOP modules for data, EDA, modeling, and MLflow
- `requirements.txt` — Python dependencies
- `main.py` — Training script for initial models (v1)
- `train_v2.py` — Training script for optimized models (v2)
- `holdout_validation.py` - Holdout evaluation script
- `artifacts` - Backups: models, metrics, feature importance

## Data
Place your `weather.csv` file in the `data/` folder.
