# Single-Cell RNA-Seq Analysis Pipeline

This project implements a pipeline for analyzing single-cell RNA sequencing (scRNA-seq) data, with a focus on using machine learning models to predict cell types.

## Key Features
- Data preprocessing and normalization using Scanpy
- Clustering with KMeans and visualization using PCA/UMAP
- Machine learning-based cell type prediction using a neural network

## Requirements
- Python 3.x
- PyTorch
- Scanpy
- Matplotlib
- scikit-learn

## Usage

1. Clone this repository:
- git clone https://github.com/MaryOlufunmilola/scRNA-seq-analysis.git
- cd single-cell-rna-seq-analysis
2. Install the required dependencies:
- pip install -r requirements.txt

3. Download the dataset:
- Dataset: [Link to dataset](https://example.com/dataset)

4. Run the preprocessing script: 
- python scripts/preprocess.py

5. Train the machine learning model:
- python scripts/machine_learning.py
