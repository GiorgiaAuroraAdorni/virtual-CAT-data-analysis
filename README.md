# virtual-CAT-data-analysis

This repository contains the code necessary for data preprocessing and analysis related to the Virtual-CAT project. 

Sensitive data has been anonymised or removed, and only the necessary code for processing and analysing the data is included. 

## Repository Structure 

The following is the directory structure of the repository:

##### Root Directory

- `run_pilot_study.sh` and `run_main_study.sh`: Shell scripts which automate the execution of data processing workflows. Specifically, they run the `data_analysis.py` script to perform data analysis and generate visualisations and tables. It specifies directories for input data and output files, including images in PDF and PNG formats, LaTeX tables, and schema files. 

##### src Directory 

- Python Scripts:
  - **plots.py**: Generates plots for visualisations.
  - **utils.py**: Contains utility functions used across scripts.
  - **data_analysis.py**: Script for performing data analysis.
  - **tables.py**: Handles table-related operations.
- **output** folder:
  - **tables**: Contains datasets for pilot and main studies.
  - **anonymised_tables**: Anonymised versions of data files.
  - **schemas**: Directory for schema files.
- **r_analysis** folder: Contains R scripts for statistical analysis and results:
  - **method.R**: Defines methods used in the analysis.
  - **statistical_analysis.R**: General statistical analysis script.
  - **results_lmm_virtual.R**: Results for virtual learning analysis.
  - **results_lmm_comparison.R**: Script for comparing linear mixed models. 

## Setup

1. **Prerequisites**

   To get started, ensure you have the following installed:

   - **Python**: Make sure you have *Python 3.9.19* installed. You can check your Python version by running:

     ```bash
     python --version
     ```

   - **R**: Ensure you have R installed. You can download it from [CRAN](https://cran.r-project.org/).

2. **Clone the Repository**
   First, clone the repository to your local machine:**

   Navigate to the `src` directory and install the required Python packages:

   ```bash
   git clone https://github.com/GiorgiaAuroraAdorni/virtual-CAT-data-analysis.git
   cd virtual-CAT-data-analysis
   ```

3. **Install Python Dependencies**

   Navigate to the `src` directory and install the required Python packages:

   ```bash
   cd src
   pip install -r requirements.txt
   ```

4. **Install R Packages**

   Open R and install the necessary R packages. Create an `install.R` script with the following content and run it:

   ```R
   install.packages(c("dplyr", "tidyverse", "broom", "emmeans", "lmerTest", "sjPlot", "car", "ggplot2", "lattice"))
   ```

## Usage

#### Running Scripts

- To execute the pilot study script:

  ```bash
  bash run_pilot_study.sh
  ```

- To execute the main study script:

  ```bash
  bash run_main_study.sh
  ```

#### Tests

- To run tests for data preparation:

  ```python
  pytest src/tests/test_data_preparation.py
  ```

