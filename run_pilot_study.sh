#!/bin/bash

cd src

# Execute the Python script for the data analysis
python data_analysis.py \
    --input_folder "output/tables/pilot/" \
    --output_folders "output/images/pilot/pdf/" "output/images/pilot/png/" \
    --tables_output_folder "output/latex/pilot/" \
    --schemas_folder "output/schemas/png/" \
    --study "pilot"
