# Online-Shoppers-Analysis
Comprehensive analysis of online shopper behavior, including EDA, dimensionality reduction with PCA, clustering (KMeans, DBSCAN), and classification (Decision Tree, Naive Bayes) to predict purchasing intention.

# Comprehensive Analysis of Online Shoppers' Intention

## Project Overview

This repository provides a comprehensive analysis of the "Online Shoppers Intention" dataset. The project aims to explore online shopper behavior, discover hidden patterns in the data, and build models to predict their purchasing intention. It covers various data science steps, including Exploratory Data Analysis (EDA), data preprocessing, dimensionality reduction, clustering, and classification.

## Files

* `main.ipynb`: The main Jupyter Notebook file containing all Python code for data analysis, visualization, preprocessing, dimensionality reduction (PCA), clustering (KMeans, DBSCAN), and building classification models (Decision Tree, Naive Bayes).
* `online_shoppers_intention.csv`: The primary dataset used for this analysis, containing various information about user sessions on an online shopping website.

## Project Steps

This project includes the following key stages:

1.  **Data Loading and Initial Inspection:**
    * Reading data from the CSV file.
    * Calculating descriptive statistics (min, max, quartiles, variance, standard deviation) for numerical features.

2.  **Exploratory Data Analysis (EDA) and Visualization:**
    * Boxplots to identify data distribution and outliers in numerical features.
    * Histograms to visualize the distribution of specific features (e.g., `Administrative_Duration`).
    * Scatter plots to observe relationships (e.g., `BounceRates` vs `ExitRates` colored by `Revenue`).
    * Pairplots to visualize pairwise relationships between selected features.
    * Heatmap to display the correlation matrix of numerical features.

3.  **Data Preprocessing:**
    * Converting categorical (object) features to numerical using Label Encoding.
    * Normalizing numerical features using `MinMaxScaler` and `StandardScaler`.

4.  **Dimensionality Reduction:**
    * Applying Principal Component Analysis (PCA) to reduce features to 2 main components, simplifying analysis and clustering.

5.  **Clustering:**
    * **KMeans:** Implementing the KMeans algorithm on PCA-reduced data, visualizing clusters, and calculating the Silhouette Score for evaluation.
    * **K-Distance Graph:** Plotting the K-Nearest Neighbor distances (for DBSCAN parameter selection).
    * **DBSCAN:** Applying the DBSCAN algorithm to identify clusters and noise points. Visualizing clusters and calculating Silhouette Score (if applicable).

6.  **Classification:**
    * Splitting data into training and testing sets.
    * **Decision Tree Classifier:** Training and evaluating a Decision Tree model with the entropy criterion.
    * **Gaussian Naive Bayes:** Training and evaluating a Naive Bayes model.
    * **Model Evaluation and Comparison:** Comparing the performance of classification models using Accuracy, Precision, and Recall metrics.

## How to Run

To run this project, follow these steps:

1.  **Clone the Repository:**
    If you haven't cloned this repository yet, do so using Git Bash or your terminal:
    ```bash
    git clone https://github.com/erfan-8/Online-Shoppers-Analysis.git
    cd Online-Shoppers-Analysis
    ```

2.  **Install Dependencies:**
    This project requires the following Python libraries:
    * `pandas`
    * `numpy`
    * `matplotlib`
    * `seaborn`
    * `scikit-learn`
    * `jupyter`

    You can install them using `pip`:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```

3.  **Run Jupyter Notebook:**
    After installing dependencies, launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
    This command will open a page in your web browser. Locate and click on the `main.ipynb` file to open the project.

4.  **View and Execute:**
    You can execute the code cells sequentially to view the data analysis steps and results.

## Data Source

The `online_shoppers_intention.csv` dataset is sourced from (Add exact source here if known, e.g., "UCI Machine Learning Repository" or "Kaggle"). It contains features from online shopping website visitor sessions and a target variable (`Revenue`) for predicting purchasing intention.

## Contribution

Suggestions and contributions to improve this project are welcome.
