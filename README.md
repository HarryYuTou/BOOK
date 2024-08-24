# Unsupervised Machine Learning Chapter

## Overview
This project explores various Unsupervised Machine Learning techniques using the "Netflix Userbase Dataset" available on Kaggle. The focus is on applying different clustering methods to uncover insights that can be used for business decisions, particularly in the context of customer segmentation.

## Chapter Objectives
- **Introduction to Unsupervised Learning**: Understand the concept of unsupervised learning and its contrast with supervised learning.
- **Clustering Methods**: Apply and compare five commonly used clustering methods:
  - K-Means Clustering
  - PAM (K-Medoids) Clustering
  - Hierarchical Clustering
  - Gaussian Mixture Model Clustering (EM algorithm)
  - Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
- **Case Study**: Use the Netflix Userbase Dataset to apply these techniques and analyze the results from a business perspective.

## Chapter Structure

### 1. **Data Preparation and Exploratory Data Analysis (EDA)**
   - **Data Loading**: Load the Netflix Userbase dataset from Kaggle.
   - **Exploratory Data Analysis (EDA)**: Analyze the dataset to understand its structure, detect any anomalies, and prepare it for clustering.

### 2. **Clustering Techniques**
   - **K-Means Clustering**:
     - Determine the optimal number of clusters using the Elbow Method.
     - Apply K-Means clustering and analyze the resulting clusters.
   - **PAM (K-Medoids) Clustering**:
     - Apply PAM clustering and compare the results with K-Means.
   - **Hierarchical Clustering**:
     - Build and analyze dendrograms using various linkage methods (Single, Average, Complete, Ward).
   - **Gaussian Mixture Model Clustering**:
     - Implement the Expectation-Maximization (EM) algorithm for clustering.
   - **DBSCAN**:
     - Explore density-based clustering and understand its application in detecting noise and outliers.
     - Use the k-NN graph to optimize DBSCAN parameters.

### 3. **Model Evaluation**
   - **Silhouette Score**:
     - Evaluate the quality of the clustering methods using the Silhouette Score.
     - Compare the performance of different clustering algorithms.

### 4. **Business Insights**
   - Analyze the clustering results to derive actionable insights for business scenarios, such as targeted marketing campaigns, customer segmentation, and recommendation systems.

## Files
- **Netflix_Data.ipynb**: The Jupyter Notebook containing the code for data analysis, clustering, and visualization.
- **Unsupervised ML.docx**: The document detailing the theory, methodology, and results of the clustering techniques applied to the Netflix Userbase dataset.

## How to Run
1. Ensure all required Python libraries are installed: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, sklearn-extra.
2. Download the Netflix Userbase dataset from Kaggle and place it in the same directory as the Jupyter Notebook.
3. Open and run the `Netflix_Data.ipynb` notebook to replicate the analysis.

## Results Summary
- **K-Means Clustering**: Found four distinct clusters based on the Elbow Plot and cluster analysis.
- **PAM (K-Medoids) Clustering**: Showed different clustering characteristics, offering an alternative perspective on the data.
- **Hierarchical Clustering**: Provided a detailed view of the cluster formation process, with Ward linkage performing the best.
- **Gaussian Mixture Model Clustering**: Highlighted overlapping clusters, useful for more complex data distributions.
- **DBSCAN**: Effective in identifying outliers, though sensitive to parameter choices.
