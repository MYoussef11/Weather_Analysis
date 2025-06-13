# Weather Regression Project

This project demonstrates an end-to-end machine learning workflow for weather data regression using Python, pandas, numpy, scikit-learn, matplotlib, seaborn, and MLflow.

## Features
- Data loading and cleaning
- Exploratory Data Analysis (EDA)
- Feature selection and preprocessing
- Model training (Linear Regression, Decision Tree, Random Forest)
- MLflow experiment tracking
- Modular, object-oriented code structure

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
   ```powershell
   python main.py
   ```

## Project Structure
- `main.py` — Entry point for the workflow
- `model/` — OOP modules for data, EDA, modeling, and MLflow
- `requirements.txt` — Python dependencies

## Data
Place your `weather.csv` file in the `data/` folder.
