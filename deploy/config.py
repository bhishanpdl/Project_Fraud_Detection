import os

# params
target = 'Class'
train_size = 0.8
test_size = 0.2
SEED = 100

# data
# https://github.com/bhishanpdl/Datasets/blob/master/Projects/Fraud_detection/raw/creditcard.csv.zip?raw=true
url_git = "https://github.com/bhishanpdl/Datasets/blob/master/Projects/"
project = "Fraud_detection"
path_data_raw_base = "/raw/creditcard.csv.zip" + "?raw=true"
path_data_raw = url_git + project + path_data_raw_base
compression = 'zip'

# model
path_model_lgb = os.path.join('.','models','clf_lgb.joblib')



