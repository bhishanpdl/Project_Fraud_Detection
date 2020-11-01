import os

# data
dat_dir = os.path.join('..','data')
data_path = os.path.join(dat_dir, 'raw/creditcard.csv.zip')
compression = 'zip'

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

# persited models
path_best_model = '../models/best_model.h5'

