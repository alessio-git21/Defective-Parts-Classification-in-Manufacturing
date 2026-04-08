# Defective Parts Classification in Manufacturing

## Project Overview

This project aims to develop a machine learning model to classify manufactured parts as 'conforming' or 'defective' based on various production parameters. The goal is to identify patterns that lead to defects, improve quality control, and reduce waste in the production process.

## Table of Contents

1.  [Data Loading and Quality Check](#1--data-loading-and-quality-check)
2.  [Descriptive Statistics](#2--descriptive-statistics)
3.  [Target Variable — Class Imbalance](#3--target-variable--class-imbalance)
4.  [Numerical Feature Distributions](#4--numerical-feature-distributions)
5.  [Correlation Analysis](#5--correlation-analysis)
6.  [Categorical Features Analysis](#6--categorical-features-analysis)
7.  [Temporal Analysis](#7--temporal-analysis)
8.  [Dataset Preparation](#8--dataset-preparation)
9.  [Model Evaluation Metrics](#9--model-evaluation-metrics)
10. [Baseline Definition](#10--baseline-definition)
11. [Logistic Regression Model](#11--logistic-regression-model)
12. [SVM with RBF Kernel](#12--svm-with-rbf-kernel)
13. [Random Forest](#13--random-forest)
14. [Random Forest — Reduced Feature Set](#14--random-forest--reduced-feature-set)
15. [Features Engineering](#15--features-engineering)
16. [Train XGBoost and analyze features importance](#16--train-xgboost-and-analyze-features-importance)
17. [Random Forest with Top 15 Feature from XGBoost](#17--random-forest-with-top-15-feature-from-xgboost)
18. [Final Conclusions](#18--final-conclusions)

## 1 · Data Loading and Quality Check

The dataset, `parts_production_data.csv`, contains 3000 rows and 16 columns related to part production. The `production_timestamp` column was converted to datetime format, spanning from 2023-12-25 to 2025-12-19. No missing values or duplicate rows were found in the initial quality check.

## 2 · Descriptive Statistics

Numerical features include `measure_diam_mm`, `measure_length_mm`, `flatness_mm`, `torque_Nm`, `surface_roughness_Ra`, `temp_process_C`, `vibration_level`, `cycle_time_s`, and `visual_inspection_score`. Categorical features are `line_id`, `station_id`, and `operator_id`.

## 3 · Target Variable — Class Imbalance

The target variable `defect_label` shows significant class imbalance:
-   **Conforming (0)**: 2327 samples (77.57%)
-   **Defective (1)**: 673 samples (22.43%)

This results in an imbalance ratio of 3.46 (Majority / Minority), necessitating the use of appropriate evaluation metrics and techniques for imbalanced datasets.

## 4 · Numerical Feature Distributions

Visual analysis through histograms and box plots showed that the distributions of conforming and defective parts across most numerical features are not highly distinct. A Mann-Whitney U test confirmed that only a few features (`temp_process_C`, `vibration_level`, `cycle_time_s`) exhibit a statistically significant difference (p < 0.05) between the two classes.

## 5 · Correlation Analysis

*   **Pearson Correlation**: A heatmap was used to visualize linear correlations between numerical features and `defect_label`. Generally, no strong linear correlations were observed.
*   **Mutual Information**: Mutual information scores were calculated to capture both linear and non-linear relationships. Scores were relatively low, indicating that individual numerical features do not provide substantial information about the `defect_label` on their own.

## 6 · Categorical Features Analysis

*   **Defect Rate by Operational Dimension**: Bar plots illustrated defect rates across different `line_id`, `operator_id`, and `station_id`. While variations exist, most operational units show similar patterns, suggesting these features might not be highly discriminative in isolation.
*   **Cramer's V Correlation**: Cramer's V scores confirmed very weak associations between categorical features (`line_id`, `station_id`, `operator_id`) and `defect_label`.

## 7 · Temporal Analysis

*   **Weekly Defect Rate**: Analysis of the defect rate over time revealed periodic patterns. The `batch_week` extracted from `material_batch` consistently matched the `week_of_year` from `production_timestamp`.
*   **Fourier Analysis**: Fourier analysis was performed on the defect rate and other numerical features. This confirmed the presence of weekly patterns, suggesting that cyclic encoding of temporal features could be beneficial for modeling.
*   **Daily and Weekly Volume Distribution**: Histograms of daily and weekly production volumes showed general distributions.

## 8 · Dataset Preparation

Based on the EDA, the following feature engineering and selection steps were applied:

*   **Rolling Statistics**: Rolling mean and standard deviation (window `N=30`) were calculated for numerical features, lagged by one unit, to capture recent production trends.
*   **Cyclic Encoding**: `production_month`, `production_dayofweek`, `production_hour`, and `production_day` were extracted from `production_timestamp` and transformed using sine and cosine functions to capture cyclic patterns.
*   **Feature Exclusion**: Original categorical features (`line_id`, `station_id`, `operator_id`) and `production_year` were excluded due to their low discriminative power and to avoid increased model complexity.
*   **Scaling**: `MinMaxScaler` was applied to numerical and rolling features. Cyclic features were already scaled between -1 and 1.
*   **PCA Visualization**: A 2D PCA projection of the preprocessed dataset showed that conforming and defective parts are not clearly separable in this reduced dimension, hinting at the complexity of the classification task.

## 9 · Model Evaluation Metrics

Due to the class imbalance, the following metrics were prioritized for evaluating model performance, particularly for the 'Defective' class:

*   **Precision**
*   **Recall**
*   **F1-Score**
*   **ROC-AUC**

Optimal decision thresholds were determined by maximizing the F1-score.

## 10 · Baseline Definition

Two baselines were established:

*   **Always Predict Healthy**: Achieved high accuracy (0.78) but zero precision, recall, and F1-score for the defective class.
*   **Random Classifier**: Averaged over 100 simulations, showed approximately 0.22 precision, recall, and F1-score for the defective class, with an overall accuracy of 0.65.

## 11 · Logistic Regression Model

A Logistic Regression model with `class_weight='balanced'` was trained. Performance metrics were poor, with an F1-score of 0.33 for the defective class and an ROC-AUC of 0.5302, slightly better than random, but indicating poor linear separability.

## 12 · SVM with RBF Kernel

An SVM with an RBF kernel and `class_weight='balanced'` yielded similar poor results, with an F1-score of 0.29 for the defective class and an ROC-AUC of 0.5184. The non-linear capabilities of SVM did not significantly improve performance.

## 13 · Random Forest

A Random Forest Classifier, also with `class_weight='balanced'`, showed an F1-score of 0.11 for the defective class and an ROC-AUC of 0.5479. Feature importance analysis highlighted `temp_process_C`, `vibration_level`, and `cycle_time_s` as the most important features, consistent with the Mann-Whitney U test.

## 14 · Random Forest — Reduced Feature Set

Given the insights from the full Random Forest, a model was trained on a reduced feature set focusing on the most important original numerical features, their rolling statistics, and cyclic temporal features. This model achieved an F1-score of 0.25 for the defective class and an ROC-AUC of 0.5433, showing no significant improvement over the full feature set. The best threshold for maximum F1-score was 0.241, yielding a precision of 0.232 and recall of 0.895 for the defective class.

## 15 · Features Engineering

Further feature engineering was performed, including:

*   **Geometric Features**: `diam_length_ratio`, `flatness_length_ratio`, `flatness_diam_ratio`.
*   **Mechanical Features**: `torque_diam_ratio`, `torque_length_ratio`, `torque_flatness_ratio`.
*   **Physical Process Features**: `temp_cycle_ratio`, `vibration_cycle_ratio`, `torque_temp_ratio`, `roughness_temp_ratio`.
*   **Surface Quality Features**: `roughness_vibration_ratio`, `roughness_cycle_ratio`, `flatness_vibration_ratio`.
*   **Contextual Features**: Differences from mean by `line_id` and `station_id` for numerical features.
*   **Temporal Features**: `feature_diff_prev` (difference from 5-step rolling mean) for numerical features.

The dataset was re-split and scaled for further modeling.

## 16 · Train XGBoost and analyze features importance

An XGBoost classifier was trained with `scale_pos_weight` to address class imbalance. It achieved an F1-score of 0.24 for the defective class and an ROC-AUC of 0.59. Feature importance from XGBoost highlighted `temp_process_C`, `vibration_level`, and newly engineered features like `vibration_cycle_ratio` and `temp_process_C_line_diff` as crucial.

## 17 · Random Forest with Top 15 Feature from XGBoost

A Random Forest model was then trained using the top 15 features identified by XGBoost. This model showed an F1-score of 0.22 for the defective class and an ROC-AUC of 0.5309. The best threshold for max F1 was 0.178, giving a precision of 0.231 and recall of 0.965. This result indicates that while recall can be pushed high, precision remains very low due to the underlying data limitations.

## 18 · Final Conclusions

The extensive analysis and modeling efforts reveal that the current dataset's features lack sufficient discriminative power to build a robust classification model for defective parts. Prioritizing recall to avoid missing defective parts leads to an unacceptably high number of false positives (low precision), which would negatively impact production efficiency. Conversely, aiming for higher precision (e.g., with a threshold of 0.6) results in a very low recall, meaning many defective parts would be missed. The models consistently show signs of overfitting, further indicating data limitations rather than model deficiencies.

Specifically, the `visual_inspection_score` was found to be unreliable as a direct predictor of defects, performing only slightly better than random chance.

**Recommendations:**

Instead of further model tuning or trying other algorithms, it is strongly recommended that the company:

1.  **Review the Visual Inspection Process**: Investigate how the `visual_inspection_score` is assigned, as its current form is not predictive.
2.  **Identify New Measurable Features**: Explore and incorporate additional, more discriminative physical or process parameters that could truly differentiate conforming from defective parts.
3.  **Consider Time Series Forecasting**: Given the identified periodic patterns in the defect rate, a more valuable approach might be to perform a predictive analysis on the defect rate time series. This could provide proactive insights for operational adjustments, helping to mitigate future peaks in defective production.

The current features do not support reliable classification, suggesting a need for better data collection or a deeper understanding of the defect causation mechanism.


