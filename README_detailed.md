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

# Author's Comments on Imbalanced Dataset
In this project I have dealt with various usual machine learning classifiers such as `Logistic Regression`,
`Support Vector Classifier`, `Decision Tree Classifier`, `Random Forest Classifier`, `K-Nearest Neighbors Classifier` with
and without resampling methods (upsampling SMOTE and downsampling) and with and without grid search (Grid Search
and Randomized Search). All these resampling methods works only when we have label column which tells whether the transaction is fraud or not.
But in real life we have label column only for training data and we do not have label for the test set. Here we should note that
the test set is highly imbalanced, there are only very few fraud cases and rest of them are non-fraud. We don't know which
transactionsa are fraud or not and thus can not downsample or upsample. When I did all the classification modelling on 
imbalanced training and test on imbalanced test set, I got zero recall (basically model predicts everything as non-fraud). If I 
do modelling on resampled data and test on imbalanced it will also not work because we are simply violating the assumption 
of machine learning: Training and Test data must come from same distribution.

To deal with imbalanced dataset, we can not use usual machine learning techniques. However, there are some specialized machine
learning techniques we can use for imbalanced dataset. In this project I have used two of these algorithms called `Local Outliers Factor (LOF)` and `Isolation Forest`. These algorithms does not give results as good as resampling methods but they have advantage
that we can test these algorithms on imbalanced dataset. Out of 98 frauds in my test set, using `LOF` gave me 3 correct frauds and `Isolation Forest` gave me 25 correct frauds. In comparison, all of the usual machine learning methods gave me 0 correct frauds out of 98 frauds in imbalanced dataset.

**Update** After doing usual machine learning algorithms, I again did a follow up with gradient boosting methods. Scikiit learn has a simple boosting algorithm `GradientBoostingClassifier` and there are other specialized libraries just to do the boosting algorithms. Out of those libraries I used `xgboost`, `lightgbm`, and `catboost` classifiers.

**Results For Imbalanced data**  
```
After train-test split, we have total frauds = 98
0 is non-fraud, 1 is fraud.
Predicting fraud as non-fraud is serious issue. (predicting 1 as 0 is bad FN is bad)
--------------------------------------------------------------------------------------
Model                        FN (Frauds predicted not frauds)
Local Outlier Factor         95 (95 wrong out of 98) 
Isolation Forest             73 (73 wrong out of 98)
XGBoost                      28 (30 wrong out of 98)
Linear Discriminant Analysis 27 (27 wrong out of 98)
LightGBM grid search         27 (19 wrong out of 98) 
Catboost grid search         21 (21 wrong out of 98)
```


# Background of the Data
- Data Source: https://www.kaggle.com/mlg-ulb/creditcardfraud

The datasets contains transactions made by credit cards in September 2013 
by european cardholders. 

This dataset presents transactions that occurred in two days,
where we have 492 frauds out of 284,807 transactions.

The dataset is highly imbalanced, the positive class (frauds)
account for 0.172% of all transactions.

Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'.

Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning.

Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

**Assumptions**:
- Here we have data for just two days, but we assume the data is representative of the whole population of credit card transactions.
- We assume the 28 variables V1-V28 are
obtained from correct method of PCA and scaled properly.

# Business Problem
When we purchase something using credit card we do not want anybody else illegally buy stuffs from our account by fraud and want to detect the fraudulent activities.

From the given features (V1-V28, Amount, Time) we will build a model that will detect whether the transaction is fraudulent or not.

Here, Class is 1 means, fraud case and 0 means non-fraud case. Out of 1000 transactions, only 2 cases are fraud and other 998 cases are non-fraud. This means the dataset is highly imbalanced.

For the imbalanced dataset, we can not use the usual accuracy metric to assess the performance of the model. If we simply build a model to predict all the transactions as non-fraud we will get 99.8% accuracy, which looks extremely good but its totally useless here since it failed to detect a single fraud case.

We are interested in fraud case not in the accuracy of the model. So, we can use the RECALL as the metric of the model evaluation.

Recall is `Recall = TP / (TP + FN)`. Here, False Negative is detecting a fraud as non-fraud. This case is more important than detecting non-fraud as fraud so we use `Recall` instead of `Precision` where other case would be much important such as in spam email detection.

We can also use AUPRC (Area Under Precision-Recall Curve) to assess the performance of the model.

For the imbalanced dataset we can also use MCC (Matthews Correlation Coefficient)

MCC is given by:
![](images/mcc.png)

Accuracy and F1-score are given by following formula:
![](images/accuracy1.png)
![](images/f1_score1.png)

Best accuracy is 1 worst is 0.  
Best F1 score is 1 worst is 0.  
Best MCC score is +1 and worst is -1.  

Consider the following case:
```
TP = 95, FP = 5; TN = 0, FN = 0.

We get:
accuracy = 95%
F1 score = 97.44%
MCC = Undefined.
```

Another example:
```
TP = 90, FP = 4; TN = 1, FN = 5
accuracy = 91%
F1 score = 95.24%
MCC = 0.14
```

Also, we should note that F1-score depends on which class is defined as positive and which class is negative.

```
TP = 95, FP = 5; TN = 0, FN = 0 ==> F1 = 97.44
TP = 0, FP = 0; TN = 5, FN = 95 ==> F1 = 0
```


MCC does not depend on which class is defined as positive and which class is negative.

About the confusion matrix:
```python
tn,fp,fn,fp = confusion_matrix(ytest,ypreds).ravel()
recall = tp / (tp+fn)


                         Precision = TP (TP/FP)   Precision has all three P's
                         ^
                         |
              Predicted  |
Correct 0     0(TN)   1(FP)
        1     0(FN)   1(TP) ---> recall = TP / (TP + FN)   Recall has one Negative
                                 Out of all 10 last true anniversaries, how many did I recall correctly?

0 is non-fraud, non-spam
1 is fraud    , spam

# We take 1 as fraud or spam cases.
FP ==> non-spam detected as spam      (but we want to send all non-spam emails)
FN ==> fraud    detected as non-fraud (but we want to find all frauds)
```
![](images/confusion_matrix.png)
Here I am interested in the quantity False Negative, I want to make it as small as possible. (Ideally zero.) And True positive as high as possible.


# Get to know with the data (EDA)

## Class Balance
As we read in the description our dataset is heavily imbalanced. There are about 2 frauds in 1000 transactions.
![](reports/figures/class_balance_donut_plot.png)

If we run classification models (e.g. Logistic regression, Random Forest Regressor) they will wrongly classify all the cases as non-frauds and get accuracy of 99% but will fail to classify correctly any of the fraud cases.

We have two choices here, either undersample the data or over-sample the data. Most common method is undersampling. It makes the model run fast but at the cost of losing lots of lots of data points. Here in this project, if we undersample the data, we will get 492 fraud cases. If we just take random sample of another 492 non-frauds our data will have roughly 1000 samples. Note that previously we had 285,000 samples.

Another way of dealing with imbalanced dataset is oversampling. One of the popular over-sampling method is called SMOTE (Synthetically Modified Oversampling Technique).

Resampling undersampling and oversampling can be described using [this image](https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets)
![](images/resampling.png)

[SMOTE Oversampling](http://rikunert.com/SMOTE_explained) will synthesize minority classes and matches with the majority classes.
![](images/smote_example.png)

For SMOTE sampling in python we can use [imbalanced-learn library](https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html) which will oversample the minority classes and creates a large dataset. In our project we have 284k non-frauds majority class and 500 samples of fraud minority case. 284k samples of fraud cases will be synthesized using the SMOTE algorithm and finally we will have 568k samples.

## Correlations
Here we look at the how the features are correlated with the target. I found that feature V17  and V14 is negatively correlated with Target and feature V11 and V4 are positively correlated with Target. This means if we increase the value of feature V17 we will see less fraudulent cases and vice versa.
![](reports/figures/correlation_with_target.png)

Negative Corelation:  
For the Class 1 (Fraud case), we can see that feature V17 has median -5, which is less than zero.
![](reports/figures/negative_correlations_with_target.png)

Positive Corelation:  
For the Class 1 (Fraud case), we can see that feature V11 has median about +4, which is greater than zero.
![](reports/figures/positive_correlations_with_target.png)


Sometimes we see collinearity between two features. If two features are collinear, they have high correlation coefficient between. If one feature increases another feature also increases and model will have difficulty which feature feature causes the changes in Target variable. If any two features are highly correlated (eg. r > 0.9) one the feature may be dropped from the dataset. Then there is a question which feature to keep and which feature to drop. We look at the correlation both features with the Target and keep the feature having larger correlation.
The correlation plot among the PCA transformed variables V1-V28 is given below. This shows there is not much collinearity among the features.
![](reports/figures/correlation_matrix_balanced.png)

## Histograms of Features
For the linear models we have an underlying assumption that the features are independent of each others and they are all distributed like Gaussian. Histogram is one of the way inspecting wheter the features look like Gaussian distributed or not.

Here, features V1-V28 are already PCA transformed, I will not change them. But the feature `Amount` can be log transformed to make it look like more Gaussian. Also the feature `Time` can be log transformed to make it look like more Gaussian.
![](reports/figures/all_features_histogram.png)

## Temporal Feature Study
Here, in this dataset we have one temporal variable called `Time`. This gives the time elapsed in seconds after the first transaction occur. The dataset says the data is from September of 2013, but it does not say which day it is and what is the first transaction time. If I assume, the first transaction time is 7AM, then the peak of fraud activity happened after two hours (9AM) and 11 hours (6PM).
![](reports/figures/normal_and_fraud_transactions.png)

Distibution Plot of Time:
![](reports/figures/distplot_time.png)
We can clearly see the bimodal distribution, this is because we have data of two days and distribution of transactions is somewhat similar in both of the days.

## Amount vs Target
Here we study the distribution of feature `Amount` and its relation with Target.
We see that most of transactions are below $100 for both fraud and non-fraud transactions.
![](reports/figures/cat_amount_countplot_fraud.png)
![](reports/figures/cat_amount_countplot_non_fraud.png)

## Distribution Plots of Features
Our data set does not have too many features, it has 28 PCA transformed features and two usual features. We can look at the distribution plot of each features for the class distribution of fraud and non-fraud and if the distribution of these two classes are same we may want to drop this feature from the entire dataset.
![](reports/figures/distplots/distplot_V28_selected_xlim.png)
![](reports/figures/distplots/distplot_V27_selected_xlim.png)
For example, for the features V28 and V27, the class distribution are more or less same for fraud and non-fraud classes and we may drop these features.
But, when I looked at correlation of these features with Target, they have considerable correlation and I am planning to keep them for the analysis.

# Statistics
In statistics we sometimes are interested in the moments of the variables. First four degrees of moments of a random variable has names: `mean`, `variance`, `skewness`, `kurtosis` and other moments are simply called moment of order n.
![](images/moments.png)


The first order moment is called mean.
For normal distribution mean is zero and variance is 1.
![](reports/figures/statistics/mean.png)
![](reports/figures/statistics/std.png)


The third order moment is called skewness.
Normal distribution has skewness of 0.
If the skewness is less than 0, it is negatively skewed and has tail on left side. If the skewness is greater than 0, it is positively skewed and has tail on right side. In our dataset, some of the features are skewed. For example: feature V8 is negatively skewed and feature V28 is positively skewed.
![](images/skewness.png)
![](reports/figures/statistics/skew.png)


If we have skewed features, then they do not follow the Gaussian distribution,and underlying assumptions of linear regression is violated. To reduce the skewness we can do log transformation (or sqrt transformation or boxcox transformation and other transformations). Here, the features are already from PCA transformation, so I am keeping them as it is.


Kurtosis is related to the fourth moment. For Normal distribution, kurtosis has a value of 3. If kurtosis > 3 it is called lepto-kurtic and it is tall peaked and on the other hand if kurtosis is less than 3, it is called platy-kurtic and looks flat. Normal distribution has other name meso-kurtic.


In our dataset some of the features have high kurtosis values. This indicates that there might be some outliers in these features. For example feature V28 has very high kurtosis value and it also has high skewness value. It might have some outliers.
![](images/kurtosis.png)
![](reports/figures/statistics/kurtosis.png)

Look at the feature V28.
```
skewness = 11.2
kurtosis = 933.4
```

Lets look at the boxplot:
![](images/boxplot.png)
![](images/boxplot1.png)
![](reports/figures/statistics/v28_boxplot.png)
We can see that lots are possible outliers present in feature V28.
Outliers distort the model's ability to fit the data properly and sometimes if we are sure these are the true outliers, we should remove these outliers.

Now, lets see the interquartile ranges of PCA features:
![](reports/figures/statistics/iqr.png)
We can see feature V1 has large IQR range.

Let's also see the standard deviation.
![](reports/figures/statistics/std.png)
IQR looks similar to standard deviations. We can say that there might not be too many outliers.

Finally, we can also look at all the features how their boxplots looks like:
![](reports/figures/statistics/boxplot_all_features.png)

# Modelling Various Classifiers
I am mostly interested in designing a classifier that will give me the lowest
False Negative values. Or, which will give me the highest Recall. Accuracy is
not the concern here. Since our dataset is heavily imbalanced,
first I undersampled the dataset and tested various machine learning
classifier models.

**False Negative Frauds Detection using Default Classifiers for Undersampled Data**
![](reports/screenshots/FN_all_default_models_undersample.png)

Then I did grid search to find the best hyperparameters
for all these models.

**False Negative Frauds Detection using Classifiers with Grid Search for Undersampled Data**
![](reports/screenshots/FN_all_models_undersample_grid.png)

**Recall for all Classifiers with Grid Search for Undersampled Data**
![](reports/screenshots/recall_all_models_undersample_grid.png)

**Classification Report for all Classifiers with Grid Search for Undersampled Data**
![](reports/screenshots/clf_report_all_models_undersample_grid.png)

# Best Model from Grid Search for Underampled Data
I found that logistic regression is the best model for classification for this dataset.
![](reports/screenshots/cm_lr_undersample_grid.png)
![](reports/figures/cv_auroc_lr_undersample_grid.png)
![](reports/figures/roc_curves_for_all_clf.png)

# Random Forest Classifier Underampled Data
One nice property of Random Forest is that, it gives the 
most important features in the dataset. For Undersampled
dataset, I found V20 was the most important feature.
![](reports/clf_best_rfc_feature_importances.png)
![](reports/screenshots/cm_rfc_undersample_grid.png)

# Detail Study of Logistic Regression with SMOTE Oversampling
Undersampling only gives results good for undersampled test set since in machine learning modelling train and test must come same distribution. Undersampling is a good 
way to get better results if we already know the class labels. But in fraud detection we only know the fitting parameters but we dont know whether the transactions is 
normal or fraudulent in advance. So our test set must be imbalanced.

When I tested the logistic regression fitted from underampled training data, it gives ZERO recall for imbalanced test set.

We need to oversample our dataset. One of the most popular
resampling method to get oversampled data is SMOTE.
Here, I used SMOTE algorithm and get the Recall of 0.42857 for the imbalanced test set.

**Logistic Regression Model Evaluation Scalar Metrics**
![](reports/screenshots/lr_model_evaluation_scalar_metrics.png)


**Logistic Regression Model Evaluation Classification Metrics**
![](reports/screenshots/lr_model_evaluation_classification_metrics.png)

**Logistic Regression Confusion Matrix**
![](reports/screenshots/cm_lr_smote_grid.png)

# Outlier Detection Models: Isolation Forest and Local Outliers Factor (LOF)
Two of the most popular models to use in Outlier Detection are Isolation Forest
and Local Outliers Factor. These models work directly on the imbalanced
dataset and we do not have to undersample or oversample the training dataset.

One technical note for these two models is that they give class results +1
and -1 and we need to change them to 0 and 1.
```python
ypreds[ypreds == 1] = 0
ypreds[ypreds == -1] = 1
ypreds_iso = ypreds
```

Isolation Forest gave me the better result than Local Outlier Factor.
```
Isolation Forest Results
-------------------------
Total Frauds:  98
Incorrect Frauds:  73
Incorrect Percent:  74.49 %

Local Outliers Factor
-------------------------
Total Frauds:  98
Incorrect Frauds:  95
Incorrect Percent:  96.94 %
```

# Modelling with LightGBM
For the underampled data I also tested the lightGBM model.
It is fast to train and gives decent results.

```
lightGBM Results
-------------------------
Total Frauds:  98
Incorrect Frauds:  62
Incorrect Percent:  63.27 %
```

After grid search, I got smaller number of errors:
```
LightGBM Grid Search Results
-------------------------
Total Frauds:  98
Incorrect Frauds:  19
Incorrect Percent:  19.39 %
```

# Deep Learning Methods
If we have large number of samples we can also use deep learning
methods to train our model. I got the following results for keras.
```
Keras Imbalaned Data Results
-------------------------
Total Frauds:  98
Incorrect Frauds:  94
Incorrect Percent:  95.92 %

Keras Undersampled Train Test Results
-------------------------
Total Frauds:  98
Incorrect Frauds:  7
Incorrect Percent:  7.14 %
```

# Big Data Analysis method using PySpark
For the imbalanced dataset I trained differet classifiers using the pyspark
module.
I got the best result for weighted Recall for random forest classifier after a
grid search.

The results are shown below:
![](reports/screenshots/pyspark_clf_results.png)
