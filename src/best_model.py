# Import libraries
import time
time_start_notebook = time.time()
import numpy as np
import pandas as pd

# local imports
import config
import util

# random state
import os
import random
import numpy as np
import tensorflow as tf

SEED=config.SEED
np.random.seed(SEED)
os.environ['PYTHONHASHSEED']=str(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# machine learning
import functools
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# deep learning
import tensorflow
import tensorflow as tf
import keras
from keras.wrappers.scikit_learn import KerasClassifier

#=================== load the data
ifile = config.data_path
df_raw = pd.read_csv(ifile,compression='zip')

#========================== data processing
eps=0.001
cols_drop = ['Time']
df = df_raw.drop(cols_drop,axis=1)
df['Ammount'] = np.log(df.pop('Amount')+eps)

#======================== train test split
target = 'Class'
df_Xtrain,df_Xtest,ser_ytrain,ser_ytest = train_test_split(
    df.drop([target],axis=1),
    df[target],
    train_size=config.train_size,
    stratify=df[target],
    random_state=SEED)

ytrain = np.array(ser_ytrain)
ytest = np.array(ser_ytest)

#======================= normalize the data
if config.PARAMS_PROCESS['scaling'] == 'standardscaler':
    lo = config.PARAMS_PROCESS['clip_low']
    hi = config.PARAMS_PROCESS['clip_high']

    scaler = StandardScaler()
    scaler.fit(df_Xtrain)

    Xtrain = scaler.transform(df_Xtrain)
    Xtest  = scaler.transform(df_Xtest)

    # clip the values
    Xtrain = np.clip(Xtrain, lo, hi)
    Xtest = np.clip(Xtest, lo, hi)

# bias intializer for the last output layer
neg, pos = np.bincount(df_raw[target])
bias_init = np.log([pos/neg])

# class weight for imbalanced data
weight_for_0 = 1 / neg
weight_for_1 = 1 / pos
class_weight = {0: weight_for_0, 1: weight_for_1}

#===================== params and metrics
n_feats = df_Xtrain.shape[-1]
METRICS = [keras.metrics.AUC(name='auc')]
PARAMS_MODEL = config.PARAMS_MODEL
PARAMS_FIT = config.PARAMS_FIT

class_weight = {0: weight_for_0, 1: weight_for_1}
PARAMS_CLF   = {'class_weight' : class_weight}

cb_early = tf.keras.callbacks.EarlyStopping(
    monitor='auc',
    verbose=1,
    patience=PARAMS_FIT['patience'],
    mode='max',
    restore_best_weights=True)

#cb_checkpt = keras.callbacks.ModelCheckpoint("fraud_model_at_epoch_{epoch}.h5")
callbacks = [cb_early]


#===================== keras classifier
# clf.fit accepts all sequential model.fit parameters
output_bias = bias_init if config.PARAMS_MODEL['use_bias_init_last_layer'] else None
get_model = functools.partial(util.get_model,params=PARAMS_MODEL,
                metrics=METRICS,n_feats=n_feats,output_bias=output_bias)

clf = KerasClassifier(get_model,
                    batch_size=PARAMS_FIT['batch_size'],
                    epochs=PARAMS_FIT['epochs'],
                    class_weight=class_weight,
                    verbose=0)

history = clf.fit(Xtrain, ytrain,
                validation_split=PARAMS_FIT['validation_split'],
                callbacks=callbacks
                )
# save fitted model
clf.model.save(config.path_best_model)

#======================= model evaluation
ypreds = clf.predict(Xtest).flatten()
cm = sklearn.metrics.confusion_matrix(ytest, ypreds)
print()
print(cm)

#========================= time taken
time_taken = time.time() - time_start_notebook
h,m = divmod(time_taken,60*60)
print('Time taken to run whole notebook: {:.0f} hr '\
      '{:.0f} min {:.0f} secs'.format(h, *divmod(m,60)))
