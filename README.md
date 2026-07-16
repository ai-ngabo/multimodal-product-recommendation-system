<<<<<<< HEAD
# multimodal-product-recommendation-system
=======
# Multimodal Product Recommendation System

This repository supports the first assignment task: merging customer social-profile and transaction data, exploring the dataset, and training a product recommendation model.

## What is included

- A merge-and-cleaning pipeline for the two provided CSV files
- Exploratory data analysis with summary plots for distribution, outliers, and correlations
- A tabular classification model to predict the likely product category from customer and transaction features
- Structured outputs in the processed and reports folders

## Folder layout

- data/unrefined/datasets/: raw CSV files
- data/processed/: merged dataset and model metrics
- reports/eda/: charts and validation notes
- notebooks/: analysis notebook
- src/: preprocessing and merge scripts

## Recommended workflow

1. Place the raw CSV files in data/unrefined/datasets/
2. Run the merge pipeline:
   python src/2_clean_merge_data.py
3. Review the generated charts in reports/eda/
4. Use the notebook in notebooks/ for a step-by-step explanation of the analysis

>>>>>>> bronze
