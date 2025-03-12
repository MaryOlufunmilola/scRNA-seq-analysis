# This script performs clustering on the processed single-cell RNA-Seq data. It includes dimensionality reduction (PCA/UMAP) and clustering (e.g., K-means or HDBSCAN).

import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Perform dimensionality reduction (PCA)
def pca_reduction(adata, n_components=50):
    """
    Perform PCA dimensionality reduction on the processed data.
    """
    sc.tl.pca(adata, n_comps=n_components)
    return adata

# Perform clustering (KMeans)
def kmeans_clustering(adata, n_clusters=6):
    """
    Perform KMeans clustering on the PCA-reduced data.
    """
    kmeans = KMeans(n_clusters=n_clusters)
    adata.obs['kmeans_clusters'] = kmeans.fit_predict(adata.obsm['X_pca'])
    return adata

# Perform UMAP visualization
def umap_visualization(adata):
    """
    Perform UMAP dimensionality reduction and visualization.
    """
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=['kmeans_clusters'], save="_clustering.png")

def main():
    # Load processed data
    adata = sc.read("data/processed_data.h5ad")
    
    # PCA reduction
    adata = pca_reduction(adata)
    
    # KMeans clustering
    adata = kmeans_clustering(adata)
    
    # UMAP visualization
    umap_visualization(adata)

if __name__ == "__main__":
    main()
