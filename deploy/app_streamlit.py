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
from sklearn import metrics as skmetrics
from scikitplot import metrics as skpmetrics
import keras

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
target = config.target
path_data_processed_Xtest = config.path_data_processed_Xtest
path_data_ytest = config.path_data_ytest
path_best_model = config.path_best_model

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

@st.cache
def st_load_data(data_path, nrows=None):
    data = pd.read_csv(data_path, nrows=nrows)
    return data

def st_header(text):
    html = """
	<div style="background-color:tomato;"><p style="color:white;font-size:30px;">{}</p></div>
	""".format(text)
    st.markdown(html, unsafe_allow_html=True)

def home():
    """Main function """

    # Project Title
    text = "Fraud Detection Binary Classification"
    st_header(text)

    # Author
    html = """<marquee style='width: 30%; color: blue;'><b> Author: Bhishan Poudel</b></marquee>"""
    st.markdown(html, unsafe_allow_html=True)
    st_display_html(path_about_html,width=600,height=600)

    #=================== load the data
    st_header("Upload or Use Default Data")
    df_Xtest = st_load_data(path_data_processed_Xtest)
    ytest = np.loadtxt(path_data_ytest)

    # Download test sample
    df_Xtest_sample = df_Xtest.sample(10)
    if st.checkbox("Download Sample Test data"):
        st.write(get_table_download_link(df_Xtest_sample),
                unsafe_allow_html=True)

    # Upload a file
    if st.checkbox("Upload Your Test data (else full data is used.)"):
        uploaded_file_buffer = st.file_uploader("")
        df_Xtest = pd.read_csv(uploaded_file_buffer)
        st.text(f"Shape of Test Data: {df_Xtest.shape}")
        st.dataframe(df_Xtest.head())
    else:
        st.text(f"Using data: {path_data_processed_Xtest}")
        st.text(f"Shape of Test Data: {df_Xtest.shape}")
        st.dataframe(df_Xtest.head())

    # Test Data Description
    st_header("Test Data Description")

    # Show Column Names
    if st.checkbox("Show Columns Names"):
        st.write(df_Xtest.columns.tolist())

    # Model Prediction
    st_header("Model Prediction and Evaluation")
    model = keras.models.load_model(path_best_model)
    yprobs = model.predict(np.array(df_Xtest)).flatten()
    ypreds = np.array(yprobs>0.5).astype(int)

    return (model,df_Xtest,ytest,ypreds,yprobs)

def model_evaluation(model,df_Xtest,ytest,ypreds,yprobs):
    df_eval = get_binary_classification_scalar_metrics('keras',
                model,df_Xtest,ytest,ypreds,yprobs,
                desc='standard-scaling').round(4)
    st.dataframe(df_eval.T)

    # scikit-plot
    txt = """\
    This is the fraud detection model. Look at the bottom left corner of
    confusion matrix shown below. This number should be small.
    These are False Negatives (FN), which are the true frauds but the model classified them as non-frauds.
    In this fradu detection we aim to reduce the metrics: FN, Recall, and area
    under the precision-recall curve.
    """
    st.text(txt)
    ax = skpmetrics.plot_confusion_matrix(ytest,ypreds)
    st.pyplot(ax.figure)

    yprobs2d = np.c_[1-yprobs,yprobs]
    ax = skpmetrics.plot_roc(ytest,yprobs2d)
    st.pyplot(ax.figure)
    ax = skpmetrics.plot_precision_recall(ytest,yprobs2d)
    st.pyplot(ax.figure)

    # plotly plot
    st_header("Model Evaluation using Plotly")
    fig = plotly_binary_clf_evaluation('keras',model,ytest,ypreds,yprobs)
    st.plotly_chart(fig)

def st_display_html(path_html,width=1000,height=500):
    fh_report = codecs.open(path_html,'r')
    page = fh_report.read()
    stc.html(page,height=height,width=width,scrolling=True)

def get_model_architecture():
    st_header("Model Architecture")
    st.text("Please wait until the website loads. If if does not display "
            "output, click Update button in the IFrame.")

    link = "https://viscom.net2vis.uni-ulm.de/L1WaKVZ8UJIHWUCGFUD7gWlNUR0kqwi3UPIUZxUtYMjyDBf65o"
    stc.iframe(link,width=1200, height=900, scrolling=True)

if __name__ == "__main__":
    menu = ["Home","About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        model,df_Xtest,ytest,ypreds,yprobs= home()
        model_evaluation(model,df_Xtest,ytest,ypreds,yprobs)
        get_model_architecture()
    elif choice == "About":
            st_display_html('about.html',width=600,height=800)
            get_model_architecture()
