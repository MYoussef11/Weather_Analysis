import pandas as pd
import logging

class DataLoader:
    """
    Loads a CSV file into a pandas DataFrame.
    """
    def __init__(self, filepath: str):
        """
        Initialize DataLoader with the path to the CSV file.
        Args:
            filepath (str): Path to the CSV file.
        """
        self.filepath = filepath
        self.data = None

    def load(self):
        """
        Loads the CSV file into a pandas DataFrame.
        Returns:
            pd.DataFrame or None: Loaded data or None if loading fails.
        """
        try:
            self.data = pd.read_csv(self.filepath)
            logging.info(f"Data loaded. Shape: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            logging.error(f"File not found: {self.filepath}")
            self.data = None
        except pd.errors.ParserError:
            logging.error(f"Failed to parse CSV: {self.filepath}")
            self.data = None
        return self.data

    def get_data(self):
        """
        Returns the loaded data if available.
        Returns:
            pd.DataFrame or None: Loaded data or None if not loaded.
        """
        if self.data is None:
            logging.warning("Data has not been loaded yet.")
        return self.data
