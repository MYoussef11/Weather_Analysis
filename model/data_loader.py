import pandas as pd

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None

    def load(self):
        self.data = pd.read_csv(self.filepath)
        print(f"Data Loaded. Shape: {self.data.shape}")
        return self.data
