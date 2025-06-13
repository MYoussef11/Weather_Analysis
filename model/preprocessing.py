import pandas as pd
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.imputed_df = None

    def impute_missing(self):
        imputer = SimpleImputer(strategy="median")
        # Impute all columns except 'date'
        columns = self.df.columns
        if 'date' in columns:
            cols_to_impute = [col for col in columns if col != 'date']
            imputed = imputer.fit_transform(self.df[cols_to_impute])
            self.imputed_df = self.df.copy()
            self.imputed_df[cols_to_impute] = imputed
        else:
            self.imputed_df = pd.DataFrame(imputer.fit_transform(self.df), columns=columns)
        return self.imputed_df
