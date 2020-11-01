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

#================== load the model
model = keras.models.load_model(config.path_best_model)

#======================= model evaluation
yprobs = model.predict(Xtest).flatten()
ypreds = [0 if i<=0.5 else 1 for i in yprobs]
cm = sklearn.metrics.confusion_matrix(ytest, ypreds)
print()
print(cm)

#========================= time taken
time_taken = time.time() - time_start_notebook
h,m = divmod(time_taken,60*60)
print('Time taken to run whole notebook: {:.0f} hr '\
      '{:.0f} min {:.0f} secs'.format(h, *divmod(m,60)))
