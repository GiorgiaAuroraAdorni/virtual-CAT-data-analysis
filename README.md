# virtual-CAT-data-analysis

Welcome to the repository for the data analysis procedures used in two related studies: a pilot study and a subsequent main study. 
This repository contains all the relevant scripts, methodologies, and documentation for analysing the data collected using the virtual CAT platform, a tool designed to assess algorithmic thinking (AT) skills in Swiss compulsory education.
Sensitive data has been anonymised or removed, and only the necessary code for processing and analysing the data is included. 

### Citation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12805318.svg)](https://doi.org/10.5281/zenodo.12805318)

If you use the materials provided in this repository, please cite the following work:

```bibtex
   @misc{adorni_virtualCAT_2023,
     author = {Adorni, Giorgia},
     doi = {10.5281/zenodo.12805318},
     month = july,
     title = {{Virtual CAT Algorithmic Thinking Assessment: Data Analysis Procedures}},
     note = {Zenodo Software. \url{https://doi.org/10.5281/zenodo.12805318}},
     year = {2024}
   }
```

### Studies
#### Unplugged CAT
The unplugged CAT refers to the original, non-digital version of the Cross Array Task (CAT), a unique assessment tool designed to evaluate AT skills without the use of digital technology. This article [**[1]**](https://doi.org/10.1016/j.chbr.2021.100166) details the framework for designing and assessing such computational thinking (CT) activities.

#### Virtual CAT
The virtual CAT is a digital embodiment of the CAT, designed to offer a versatile platform for assessing AT in educational settings. The platform integrates various interfaces and software components to enhance the assessment process. Details of this framework are described in the upcoming software article **[2]**. The following open-source software components are integral to the virtual CAT platform [**[3]**](https://doi.org/10.5281/zenodo.10027851), [**[4]**](https://doi.org/10.5281/zenodo.10016535), [**[5]**](https://doi.org/10.5281/zenodo.10015011).

The _pilot study_ explores the application of the virtual CAT tool for assessing AT skills. This study is currently under review with the International Journal of Child-Computer Interaction **[6]**. 
The dataset supporting this pilot study is available here [**[8]**](https://doi.org/10.5281/zenodo.10018292).

The _main study_ compares the effectiveness of the virtual CAT to the original unplugged CAT for assessing AT skills. This study is currently under review with Computers in Human Behavior Reports **[7]**.
The dataset for this main study is available here [**[9]**]([https://doi.org/10.5281/zenodo.10018292](https://doi.org/10.5281/zenodo.10912339)).


##### REFERENCES

**[1]** Piatti, A., Adorni, G., El-Hamamsy, L., Negrini, L., Assaf, D., Gambardella, L., & Mondada, F. (2022). The CT-cube: A framework for the design and the assessment of computational thinking activities. Computers in Human Behavior Reports, 5, 100166. https://doi.org/10.1016/j.chbr.2021.100166

**[2]** Adorni, G., Piatti, S., & Karpenko, V. (2024). Virtual CAT: A Multi-Interface Educational Platform for Algorithmic Thinking Assessment. Accepted at SoftwareX on April 11, 2024.

**[3]** Adorni, G., & Piatti, S., & Karpenko, V. (2023). virtual CAT: An app for algorithmic thinking assessment within Swiss compulsory education. Zenodo Software. https://doi.org/10.5281/zenodo.10027851 On GitHub: https://github.com/GiorgiaAuroraAdorni/virtual-CAT-app/

**[4]** Adorni, G., & Karpenko, V. (2023). virtual CAT programming language interpreter. Zenodo Software. https://doi.org/10.5281/zenodo.10016535 
On GitHub: https://github.com/GiorgiaAuroraAdorni/virtual-CAT-programming-language-interpreter/

**[5]** Adorni, G., & Karpenko, V. (2023). virtual CAT data infrastructure. Zenodo Software. https://doi.org/10.5281/zenodo.10015011
On GitHub: https://github.com/GiorgiaAuroraAdorni/virtual-CAT-data-infrastructure

**[6]** Adorni, G., & Piatti, A. (2024). The Virtual CAT: A Tool for Algorithmic Thinking Assessment in Swiss Compulsory Education. Submitted October 27, 2023 to International Journal of Child-Computer Interaction.

**[7]** Adorni, G., Artico, I., Piatti, A., Lutz, E., Gambardella, L. M., Negrini, L., Mondada, F., & Assaf, D. (2024). Assessing Algorithmic Skills in Compulsory Education: A Comparative Study of Plugged and Unplugged Approaches. Submitted June 10, 2024 to Computers in Human Behavior Reports.

**[8]** Adorni, G. (2023). Dataset from the pilot study of the virtual CAT platform for algorithmic thinking skills assessment in Swiss Compulsory Education. Zenodo Dataset. https://doi.org/10.5281/zenodo.10018292

**[9]** Adorni, G. (2023). Dataset from the main large scale study of the virtual CAT platform for algorithmic thinking skills assessment in Swiss Compulsory Education. Zenodo Dataset. https://doi.org/10.5281/zenodo.10912339 


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

