# Performance Evaluation of Data Models

## Overview
This project explores the performance evaluation of data models, emphasizing k-nearest neighbors (KNN) and Naïve Bayes algorithms. The primary datasets used are `Diabetes.csv` for the KNN model and `Iris.csv` for the GaussianNB model, a variant of Naïve Bayes.

## K-Nearest Neighbors Model
### Dataset: Diabetes.csv
- **Objective**: Create a k-nearest neighbors model to predict diabetes and assess its performance.
- **Methodology**:
  - Comparison of `y_test` (actual values) with `y_pred` (predicted values).
  - Use of `sklearn.metrics` library to compute performance measures, including precision, recall, and F-measure.
  - Evaluation of the model's performance on both training and testing datasets.

## Naïve Bayes Model
### Dataset: Diabetes.csv
- **Objective**: Develop a Naïve Bayes model using the same dataset and compare its performance against the KNN model.

## GaussianNB Model
### Dataset: Iris.csv
- **Objective**: Implement a GaussianNB model focused on flower prediction.
- **Tasks**:
  - Perform predictions related to different flower types in the Iris dataset.
  - Provide a comprehensive dataset description and analysis.

## Discussion
- **Laplace Correction in Naïve Bayes**: 
  - This correction is crucial for avoiding zero probabilities in the model, thereby enhancing prediction reliability.

## Conclusion
The project aims to provide a comparative analysis of KNN and Naïve Bayes models in terms of performance metrics, offering insights into their applicability and effectiveness in different scenarios.
