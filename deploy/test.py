# Imports
import numpy as np
import pandas as pd
import os
import time

# internet
import urllib
import codecs
import base64

# visualization
import matplotlib.pyplot as plt

# modelling
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics as skmetrics
from scikitplot import metrics as skpmetrics
import joblib

# local imports
import config
from util_model_eval import (get_binary_classification_scalar_metrics,
                            plotly_binary_clf_evaluation)

# model deploy
import streamlit as st
import streamlit.components.v1 as stc

# settings
st.set_option("deprecation.showfileUploaderEncoding", False)

__doc__ = """
Date   : Nov 19, 2020
Author : Bhishan Poudel
Purpose: Interactive report of the project

Command:
streamlit run app_streamlit.py
"""

# Parameters
SEED = config.SEED
target = config.target
compression = config.compression
path_data_raw = config.path_data_raw
path_model_lgb = config.path_model_lgb

path_about_html = "deploy/about.html"

def get_table_download_link(df, filename='data.csv', linkname='Download Data'):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{linkname}</a>'
    return href

def st_header(text):
    html = """
	<div style="background-color:tomato;"><p style="color:white;font-size:30px;">{}</p></div>
	""".format(text)
    st.markdown(html, unsafe_allow_html=True)

@st.cache
def st_load_data(data_path, nrows=None,compression=compression):
    df = pd.read_csv(data_path, nrows=nrows,compression=compression)
    df_Xtrain, df_Xtest, ser_ytrain, ser_ytest = train_test_split(
    df.drop(target,axis=1),
    df[target],
    test_size=0.2,
    random_state=SEED,
    stratify=df[target])

    return (df_Xtrain,df_Xtest,ser_ytrain,ser_ytest)

#============================================================

df_Xtrain,df_Xtest,ser_ytrain,ser_ytest = st_load_data(path_data_raw)
ytest = np.array(ser_ytest).flatten()
