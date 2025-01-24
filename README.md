
# Clustering Algorithms for Numerical and Categorical Data

This repository contains Python implementations of various clustering algorithms and utilities for processing numerical and categorical data. These algorithms were developed in 2019 to address clustering challenges in machine learning tasks, offering versatile methods for both data preprocessing and evaluation.
## Table of Contents

- [Features](#features)
- [Usage](#usage)
- [Algorithms and Functions](#algorithms-and-functions)

## Features

- **Clustering Methods**:
  - Squeezer for categorical data
  - K-Prototypes for mixed data
  - Numerical clustering with various algorithms (KMeans, DBSCAN, Gaussian Mixture, etc.)
- **Evaluation Metrics**:
  - Category Utility (CU)
  - Variance of Clusters (VAR)
  - Cluster Validation using CU and Variance
- **Preprocessing Utilities**:
  - Encoding categorical data
  - Converting CSV to TSV

## Usage

### Example Workflow

```python
import pandas as pd
from your_module import squeezer_cluster, VAR, category_utility

# Load dataset
data = pd.read_csv('example.csv')

# Run Squeezer clustering
clusters, num_clusters = squeezer_cluster(data, threshold=7)
print(f"Number of clusters: {num_clusters}")

# Compute Variance of clusters
variance = VAR(raw_data=data, clustering=clusters)
print(f"Cluster Variance: {variance}")

# Calculate Category Utility
cu = category_utility(raw_data=data, clustering=clusters, m=3)
print(f"Category Utility: {cu}")
```

## Algorithms and Functions

### Clustering Algorithms

1. **Squeezer**:
   - Designed for categorical data.
   - Dynamically forms clusters based on a similarity threshold.

2. **K-Prototypes**:
   - Handles mixed data types (numerical + categorical).
   - Requires specifying categorical feature indices.

3. **Numerical Clustering**:
   - Implements several numerical clustering algorithms like KMeans, DBSCAN, and Gaussian Mixture.

### Evaluation Metrics

- **Category Utility (CU)**: Measures the quality of clusters by evaluating intra-cluster similarity and inter-cluster differences.
- **Variance (VAR)**: Computes the variance of clusters for numerical data.

### Utility Functions

- `csv_to_tsv`: Converts CSV files to TSV format.
- `cat_utility`: Computes category utility for a clustering.
- `VAR`: Calculates the variance of clusters.
- `CV`: Combines cluster utility and variance.

