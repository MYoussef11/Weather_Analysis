import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.features = None
        self.target = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.X_train_scaled = self.X_test_scaled = None
        self.scaler = None

    def select_features(self):
        # Use all columns except 'date' and 'mean_temp' as features
        self.features = self.df.drop(columns=['date', 'mean_temp'], errors='ignore')
        self.target = self.df['mean_temp']
        return self.features, self.target

    def split(self, train_size=0.7, random_state=0):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, train_size=train_size, random_state=random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale(self):
        self.scaler = StandardScaler()
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train), columns=self.X_train.columns
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test), columns=self.X_test.columns
        )
        return self.X_train_scaled, self.X_test_scaled
