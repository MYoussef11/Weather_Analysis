import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def summary(self):
        print("Summary statistics:")
        print(self.df.describe())

    def plot_mean_temp_distribution(self):
        if 'mean_temp' in self.df.columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.df['mean_temp'], bins=30, kde=True)
            plt.title("Distribution of Mean Temperature")
            plt.xlabel("Mean Temperature (°C)")
            plt.show()
        else:
            print("'mean_temp' column not found.")
