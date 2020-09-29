Table of Contents
=================
   * [Background of the Data](#background-of-the-data)
   * [Business Problem](#business-problem)
   * [Get to know with the data (EDA)](#get-to-know-with-the-data-eda)
      * [Class Balance](#class-balance)
      * [Correlations](#correlations)
      * [Histograms of Features](#histograms-of-features)
      * [Temporal Feature Study](#temporal-feature-study)
      * [Amount vs Target](#amount-vs-target)
      * [Distribution Plots of Features](#distribution-plots-of-features)
   * [Statistics](#statistics)
   * [Modelling Various Classifiers](#modelling-various-classifiers)
   * [Best Model from Grid Search for Underampled Data](#best-model-from-grid-search-for-underampled-data)
   * [Random Forest Classifier Underampled Data](#random-forest-classifier-underampled-data)
   * [Detail Study of Logistic Regression with SMOTE Oversampling](#detail-study-of-logistic-regression-with-smote-oversampling)
   * [Outlier Detection Models: Isolation Forest and Local Outliers Factor (LOF)](#outlier-detection-models-isolation-forest-and-local-outliers-factor-lof)
   * [Modelling with LightGBM](#modelling-with-lightgbm)
   * [Deep Learning Methods](#deep-learning-methods)
   * [Big Data Analysis method using PySpark](#big-data-analysis-method-using-pyspark)

# Project Description
In this project I used the [Kaggle Creditcard Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
data to determine whether the transaction is fraud or not.


**Assumptions**:
- Here we have data for just two days, but we assume the data is representative of the whole population of credit card transactions.
- We assume the 28 variables V1-V28 are
obtained from correct method of PCA and are scaled properly.

**Metric Used**
- Here 1 means fraud and 0 means no fraud.
- The metric used is `Recall = TP / (TP + FN)` since we are interested in fraud case, not the overall accuracy of the model.

**Resampling Techniques**
- Under-sampling. (We have low number of frauds, choose randomly same number of non-frauds.)
- Oversampling `SMOTE` method. Used external library `imblearn`.

# Undersampling
**Recall for all Classifiers with Grid Search for Undersampled Data**
![](reports/screenshots/recall_all_models_undersample_grid.png)
![](reports/screenshots/cm_lr_undersample_grid.png)

# SMOTE Oversampling: Logistic Regression
![](reports/screenshots/lr_model_evaluation_scalar_metrics.png)
![](reports/screenshots/cm_lr_smote_grid.png)

# Anomaly Detection Methods
| Model | Description | Accuracy | Precision | Recall | F1(Weighted) |
| :---|:---|:---|:---|:---|:---|
| Isolation Forest | default | 0.997384 | 0.261682 | 0.285714 | 0.997442 |
| Local Outlier Factor | default | 0.996331 | 0.025641 | 0.030612 | 0.996493 |

# Gradient Boosting Modelling: LightGBM
| Model | Description | Accuracy | Precision | Recall | F1 | AUC |
| :---|:---|:---|:---|:---|:---|:---|
| lightgbm | grid search optuna | 0.999315 | 0.873418 | 0.704082 | 0.779661 | 0.851953 |
| lightgbm | default | 0.997367 | 0.275862 | 0.326531 | 0.299065 | 0.662527 |

# Deep Learning: Simple keras model


# Big Data Modelling: PySpark
![](reports/screenshots/pyspark_clf_results.png)
