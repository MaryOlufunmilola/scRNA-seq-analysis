# This script handles the preprocessing and normalization of single-cell RNA-Seq data. It includes filtering genes, normalizing the data, and identifying variable genes.
import scanpy as sc
import pandas as pd
import numpy as np

# Load raw data
def load_data(file_path):
    """
    Load the raw single-cell RNA-Seq data (CSV, TSV, etc.).
    """
    data = pd.read_csv(file_path, index_col=0)
    return data

# Preprocess data
def preprocess_data(adata):
    """
    Preprocesses the data: normalization, filtering, and identifying variable genes.
    """
    # Filter genes and cells with low expression
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Normalize the data to the total count per cell and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Identify highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]

    return adata

# Save processed data
def save_data(adata, output_file):
    """
    Save the processed AnnData object.
    """
    adata.write(output_file)

def main():
    # Load data
    file_path = "data/pbmc3k_data.csv"  # Replace with actual file path
    adata = sc.read(file_path)
    
    # Preprocess data
    adata = preprocess_data(adata)
    
    # Save the processed data
    output_file = "data/processed_data.h5ad"
    save_data(adata, output_file)

if __name__ == "__main__":
    main()

