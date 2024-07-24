#!/bin/bash

cd src

# Execute the Python script for the data analysis
python data_analysis.py \
    --input_folder "output/tables/main/" \
    --output_folders "output/images/main/pdf/" "output/images/main/png/" \
    --tables_output_folder "output/latex/main/" \
    --schemas_folder "output/schemas/png/" \
    --study "main"
