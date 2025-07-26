# Psychology Data Analysis Agents

This repository contains a collection of simple agents created for a psychology major
with a statistics minor.  Each agent automates common data analysis tasks that
are frequently encountered in psychological research.

## 1. Survey Data Analyzer

The Survey Data Analyzer helps you load survey or experimental data from a CSV
file, compute descriptive statistics, examine correlations between numeric
variables, and generate basic histograms for selected variables.

### Features

* **Data loading:** Reads a CSV file into a pandas DataFrame.
* **Descriptive statistics:** Calculates count, mean, standard deviation, min,
  quartiles, and max for numeric variables.
* **Correlation matrix:** Produces a Pearson correlation matrix for numeric variables.
* **Histogram plotting:** Generates histograms of selected numeric variables and
  saves them as PNG files.

### Usage

1. Place your CSV data in the same directory as `survey_analyzer.py`.
2. Run the script from the command line:

   ```bash
   python survey_analyzer.py
   ```

   By default it will load the provided `active.csv` dataset.  You can modify
   the `sample_file` variable in the `main()` function to point to your own data.
3. Inspect the console output for descriptive statistics and the correlation matrix.
4. View the generated histograms in the `plots` directory.

## 2. Future Agents

This repository is intended to grow.  Additional agents could include:

* **Literature Review Summarizer:** Parses academic papers to extract key findings.
* **Experimental Design Planner:** Provides sample sizes and experimental designs based on your parameters.
* **Data Visualization Assistant:** Automatically generates more complex plots (boxplots, scatter plots, etc.).

Feel free to fork this repository and add your own agents.  Each new agent should
reside in its own module with a clear description and usage instructions.

---

*This portfolio demonstrates your ability to automate data processing tasks in
Python.  When sharing on LinkedIn, consider linking to the repository and
highlighting specific agents in your posts or profile.*