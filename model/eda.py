import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class EDA:
    """
    Exploratory Data Analysis (EDA) for weather data.
    Provides summary statistics, distribution plots, and correlation heatmaps.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize EDA with a DataFrame.
        Args:
            df (pd.DataFrame): DataFrame to analyze.
        """
        self.df = df

    def summary(self):
        """
        Print summary statistics of the DataFrame.
        """
        logging.info("Summary statistics:")
        print(self.df.describe())

    def missing_data_summary(self):
        """
        Print the percentage of missing values per column.
        """
        missing_percent = self.df.isnull().mean() * 100
        logging.info("Missing data summary:")
        print("% Missing per column:\n", missing_percent)

    def boxplots(self, columns=None):
        """
        Plot each numeric feature's boxplot separately in a vertical grid, with auto-zoom for each feature.
        Args:
            columns (list, optional): List of columns to plot. If None, plots all numeric columns.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['float', 'int']).columns
        n = len(columns)
        fig, axes = plt.subplots(n, 1, figsize=(8, 5 * n))
        if n == 1:
            axes = [axes]
        for i, col in enumerate(columns):
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            sns.boxplot(x=self.df[col], ax=axes[i], orient="v")
            axes[i].set_title(f"Boxplot of {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylim(lower, upper)
        plt.tight_layout()
        plt.show()
        logging.info(f"Boxplots plotted for columns: {list(columns)}.")

    def plot_mean_temp_distribution(self):
        """
        Plot the distribution of mean temperature.
        """
        if 'mean_temp' in self.df.columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.df['mean_temp'], bins=30, kde=True)
            plt.title("Distribution of Mean Temperature")
            plt.xlabel("Mean Temperature (°C)")
            plt.show()
        else:
            logging.warning("'mean_temp' column not found.")

    def plot_correlation_heatmap(self):
        """
        Plot a correlation heatmap for the DataFrame features.
        """
        try:
            plt.figure(figsize=(10, 8))
            corr = self.df.corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Feature Correlation Heatmap")
            plt.show()
            logging.info("Correlation heatmap plotted successfully.")
        except Exception as e:
            logging.error(f"Error plotting correlation heatmap: {e}")
