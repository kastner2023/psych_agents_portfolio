"""
survey_analyzer.py
====================

This module implements a simple SurveyDataAnalyzer class that can be used
to load a CSV file containing survey responses, compute descriptive
statistics, calculate correlation matrices, and produce basic visualizations.

The class is intentionally lightweight and beginner‑friendly.  It uses
well‑known Python libraries (pandas, numpy, matplotlib) to perform the
underlying computations.  Psychology majors with a statistics minor can
extend this class further to compute other psychometric properties (e.g.,
reliability coefficients) or to build more advanced plots.

Example usage::

    from survey_analyzer import SurveyDataAnalyzer
    analyzer = SurveyDataAnalyzer('active.csv')
    analyzer.load_data()
    print(analyzer.describe())
    print(analyzer.correlation_matrix())
    analyzer.plot_histograms(['age', 'mmse'])

The included ``main`` block demonstrates how to run the analyzer on a
sample file provided in the repository (``active.csv``).  When run as a
script, the module will load that dataset, print a summary, display the
correlation matrix, and save histograms for a few selected variables.

Dependencies:

* pandas
* numpy
* matplotlib

These libraries are standard in many Python distributions and should be
available in most data science environments.  If they're missing, you can
install them via pip::

    pip install pandas numpy matplotlib

"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
import numpy as np
import matplotlib

# Use a non‑interactive backend for matplotlib so that plots can be saved
matplotlib.use('Agg')  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class SurveyDataAnalyzer:
    """A helper class for loading and analyzing survey data stored in CSV files.

    Attributes
    ----------
    file_path : str
        Path to the CSV file containing survey responses.
    df : Optional[pd.DataFrame]
        The loaded data frame.  Initially set to ``None`` until ``load_data``
        is called.

    """

    file_path: str
    df: Optional[pd.DataFrame] = field(init=False, default=None)

    def load_data(self) -> pd.DataFrame:
        """Load the CSV file into a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The loaded data.

        Raises
        ------
        FileNotFoundError
            If the file specified by ``file_path`` does not exist.
        pd.errors.EmptyDataError
            If the file is empty.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file '{self.file_path}' does not exist.")
        self.df = pd.read_csv(self.file_path)
        return self.df

    def describe(self) -> pd.DataFrame:
        """Return descriptive statistics for numerical columns in the dataset.

        This method requires that ``load_data`` has been called.  It
        computes count, mean, standard deviation, minimum, quartiles and
        maximum for each numeric column.

        Returns
        -------
        pd.DataFrame
            A summary of descriptive statistics.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        # Select only numeric columns
        numeric_df = self.df.select_dtypes(include=[np.number])
        return numeric_df.describe()

    def correlation_matrix(self) -> pd.DataFrame:
        """Compute the Pearson correlation matrix between numeric variables.

        Returns
        -------
        pd.DataFrame
            A correlation matrix for numeric columns.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        numeric_df = self.df.select_dtypes(include=[np.number])
        return numeric_df.corr()

    def plot_histograms(self, variables: List[str], bins: int = 20, output_dir: str = 'plots') -> List[str]:
        """Generate histograms for selected variables and save them to disk.

        Parameters
        ----------
        variables : list of str
            Names of the columns to plot.  Non‑numeric variables will be
            ignored with a warning.
        bins : int, default=20
            Number of bins to use for the histograms.
        output_dir : str, default='plots'
            Directory where the plot images will be saved.  It will be created
            if it does not already exist.

        Returns
        -------
        list of str
            A list of file paths to the saved histogram images.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        for var in variables:
            if var not in self.df.columns:
                print(f"Warning: Column '{var}' not found in the dataset; skipping.")
                continue
            # Only plot numeric variables
            if not np.issubdtype(self.df[var].dtype, np.number):
                print(f"Warning: Column '{var}' is not numeric; skipping.")
                continue
            plt.figure()
            plt.hist(self.df[var].dropna(), bins=bins, edgecolor='black')
            plt.title(f"Histogram of {var}")
            plt.xlabel(var)
            plt.ylabel('Frequency')
            file_name = f"{var}_hist.png"
            file_path = os.path.join(output_dir, file_name)
            plt.savefig(file_path)
            plt.close()
            saved_paths.append(file_path)
        return saved_paths


def main():
    """Demonstrate how to use the SurveyDataAnalyzer on a sample dataset.

    This function loads ``active.csv``, prints descriptive statistics and
    correlation matrix, and saves histograms for a few selected variables.
    """
    # Default file path pointing to the included sample dataset
    sample_file = os.path.join(os.path.dirname(__file__), 'active.csv')
    analyzer = SurveyDataAnalyzer(sample_file)
    df = analyzer.load_data()
    print("Loaded data with shape:", df.shape)
    print("\nDescriptive Statistics:\n", analyzer.describe())
    print("\nCorrelation Matrix:\n", analyzer.correlation_matrix())
    # Choose a few variables to plot (modify as needed)
    hist_vars = ['age', 'edu', 'mmse']
    paths = analyzer.plot_histograms(hist_vars, bins=15)
    print(f"Saved histogram plots to: {paths}")


if __name__ == '__main__':
    main()