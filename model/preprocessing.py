import pandas as pd

class DataPreprocessor:
    """
    Handles feature selection from a pandas DataFrame for weather data.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataPreprocessor with a DataFrame.
        Args:
            df (pd.DataFrame): Input DataFrame to preprocess.
        """
        self.df = df.copy()

    def select_features(self):
        """
        Select features and target from the DataFrame, ensuring no NaNs in target.
        Returns:
            features (pd.DataFrame): Feature columns.
            target (pd.Series): Target column (mean_temp).
        """
        # Drop rows where mean_temp is NaN
        df_clean = self.df.dropna(subset=['mean_temp'])
        features = df_clean.drop(columns=['date', 'mean_temp'], errors='ignore')
        target = df_clean['mean_temp']
        # Optional safety check
        assert target.isna().sum() == 0, "Target y still contains NaNs"
        return features, target

