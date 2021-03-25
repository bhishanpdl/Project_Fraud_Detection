# Project Description
In this project I used the [Kaggle Creditcard Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
data to determine whether the transaction is fraud or not.


**Assumptions**:  
- Here we have data for just two days, but we assume the data is representative
 of the whole population of credit card transactions.
- We assume the 28 variables V1-V28 are
obtained from correct method of PCA and are scaled properly.

**Metric Used**  
- Here 1 means fraud and 0 means no fraud.
- For this imbalanced dataset, false negative (fraud classified as not fraud) is more important than false negative, so we use `Recall` as the metric of evaluation. (`Recall = TP / (TP + FN)`).
- For the imbalanced dataset, AUCROC gives overly optimistic metric, instead we should use `precision_recall_curve` and after looking at the curve we should choose the value that we want for precision and recall.
- We should also note that precision and recall does not involve TN, so we should use them only when specificity (TNR = TN/(TN+FP)) is not important.
- For imbalanced dataset, we can use F_beta metric. If both precision and recall are equally important, we can use F1-score. If we consider recall beta times more important than precision, we can use `F_beta = (1+beta^2) PR/(beta^P + R)` where P is precision and R is recall. (Mnemonic: Look at the denominator and remember that Recall is beta^2 time important than Precision). (Common values are 2 and 0.5. If beta is 2, recall is twice important than precision.)
- We should also note that F_beta depends on Precision and Recall only. It does not depend on TN (true negative), so for imbalanced classification, better metric could be MCC (Mathew's Correlation Coefficient.)

**Resampling Techniques**  
- Our dataset is imbalanced, we can try two sampling: undersampling and oversampling.
- Under-sampling. (We have low number of frauds, choose randomly same number of non-frauds.)
- Oversampling `SMOTE` method. Used external library `imblearn`.

<h1 style="background-color:tomato;">Best Model So Far</h1>

| Model | Description | Accuracy | Precision | Recall | F1 | AUC | Untrue Frauds| Missed Frauds|
| :---|:---|:---|:---|:---|:---|:---|:---|:---|
|keras|	1 layer, class_weight, early_stopping, scikit api|	0.987939|	0.111989|	0.867347|	0.198366|	0.927747| 674 | 13|
| cb_tuned pycaret | fold=5 | 0.9996 | 0.9659 | 0.7865 | 0.9667 | 0.8642 | | |
|catboost |	seed=100,depth=6,iter=1k|	0.999631|	1.000000|	0.785714|	0.880000|	0.892857|0 | 21|

</br>

<h1 style="background-color:tomato;">Undersampling</h1>

**Recall for all Classifiers with Grid Search for Undersampled Data**
![](reports/screenshots/recall_all_models_undersample_grid.png)
![](reports/screenshots/cm_lr_undersample_grid.png)

</br>

<h1 style="background-color:tomato;">SMOTE Oversampling: Logistic Regression</h1>

![](reports/screenshots/lr_model_evaluation_scalar_metrics.png)
![](reports/screenshots/cm_lr_smote_grid.png)

</br>

<h1 style="background-color:tomato;">Anomaly Detection Methods</h1>

| Model | Description | Accuracy | Precision | Recall | F1(Weighted) |
| :---|:---|:---|:---|:---|:---|
| Isolation Forest | default | 0.997384 | 0.261682 | 0.285714 | 0.997442 |
| Local Outlier Factor | default | 0.996331 | 0.025641 | 0.030612 | 0.996493 |

</br>

<h1 style="background-color:tomato;">Gradient Boosting Modelling</h1>

| Model | Description | Accuracy | Precision | Recall | F1 | AUC |
| :---|:---|:---|:---|:---|:---|:---|
| lightgbm | grid search optuna | 0.999315 | 0.873418 | 0.704082 | 0.779661 | 0.851953 |
| lightgbm | default | 0.997367 | 0.275862 | 0.326531 | 0.299065 | 0.662527 |
| Xgboost | default, imbalanced | 0.999263 | 0.850000 | 0.693878 | 0.764045 | 0.846833 |
| Xgboost | default, undersampling | 0.999263 | 0.850000 | 0.693878 | 0.764045 | 0.846833 |
| Xgboost | n_estimators=150, imbalanced | 0.999263 | 0.850000 | 0.693878 | 0.764045 | 0.846833 |
| Xgboost | undersample, hpo1 | 0.999298 | 0.881579 | 0.683673 | 0.770115 | 0.841758 |
| Xgboost | imbalanced, hpo | 0.999245 | 0.898551 | 0.632653 | 0.742515 | 0.816265 |
| xgboost | grid search optuna | 0.999333 | 0.875000 | 0.714286 | 0.786517 | 0.857055 |
|catboost |	seed=100,depth=6,iter=1k|	0.999631|	1.000000|	0.785714|	0.880000|	0.892857|


</br>

<h1 style="background-color:tomato;">Automatic Modelling: pycaret</h1>

| Model | Description | Accuracy | AUC | Recall | Precision | F1 | Kappa |
| :---|:---|:---|:---|:---|:---|:---|:---|
| cb_tuned | fold=5 | 0.9996 | 0.9659 | 0.7865 | 0.9667 | 0.8642 | 0.8639 |
| lda_tuned | fold=5 | 0.9995 | 0.9833 | 0.7760 | 0.9217 | 0.8423 | 0.8420 |
| xgb | default | 0.9994 | 0.9585 | 0.7345 | 0.9102 | 0.8047 | 0.8044 |
| cb | default | 0.9995 | 0.9554 | 0.7345 | 0.9548 | 0.8215 | 0.8212 |
| lda | default | 0.9992 | 0.9677 | 0.7255 | 0.8340 | 0.7661 | 0.7657 |
| xgb_tuned | tuned | 0.9992 | 0.9677 | 0.7255 | 0.8340 | 0.7661 | 0.7657 |
| lda_tuned | n_iter=100,fold=10 | 0.9992 | 0.9677 | 0.7255 | 0.8340 | 0.7661 | 0.7657 |

</br>

<h1 style="background-color:tomato;">Big Data Modelling: PySpark</h1>

![](reports/screenshots/pyspark_clf_results.png)

</br>

<h1 style="background-color:tomato;">Deep Learning Models</h1>

| Model | Description | Accuracy | Precision | Recall | F1 | AUC | Missed Frauds| Untrue Frauds|
| :---|:---|:---|:---|:---|:---|:---|:---|:---|
|keras|	3 layers, 2 dropouts, class_weight| 0.983744|	0.081818|	0.826531|	0.148897|	0.905273| 17 | 909|
|keras|	1 layer, dropout, early_stopping|	0.984990|	0.090811|	0.857143|	0.164223|	0.921177| 14| 841|
|keras|	1 layer, dropout, steps_per_epoch, oversampling|	0.982796|	0.080000|	0.857143|	0.146341|	0.920077|14 | 966 |
|keras|	1 layer, class_weight, early_stopping, scikit api|	0.987939|	0.111989|	0.867347|	0.198366|	0.927747| 13 | 674|


# References
- https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
- https://keras.io/examples/structured_data/imbalanced_classification/
- https://www.kaggle.com/residentmario/using-keras-models-with-scikit-learn-pipelines#
- https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
