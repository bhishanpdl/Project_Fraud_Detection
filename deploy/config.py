import os

# target
target = 'Class'

# data
dat_dir = os.path.join('.','data')
url = "https://github.com/bhishanpdl/Datasets/blob/master/Projects/Fraud_detection/processed/processed_Xtest.csv?raw=true"

path_data_processed_Xtest = os.path.join(url)

path_data_ytest = os.path.join(dat_dir,'processed/ytest.csv')
path_best_model = os.path.join('.','models','best_model.h5')


# params
train_size = 0.8
SEED = 100

# params
PARAMS_PROCESS = {
    'scaling': 'standardscaler',
    'clip_low': -5,
    'clip_high': 5

}

# params model must follow names such as L1_units, L1_act, L1_dropout
PARAMS_MODEL = {
    # layer 1
    'L1_units': 16,
    'L1_act': 'relu',
    'L1_dropout': 0.5,

    # optimizer
    'adam_lr': 1e-3,

    # bias initializer
    'use_bias_init_last_layer': True,
}

PARAMS_FIT = {
    'epochs': 100,
    'batch_size': 2048,
    'patience': 10,
    'validation_split': 0.2,
    'shuffle': True,
    }
