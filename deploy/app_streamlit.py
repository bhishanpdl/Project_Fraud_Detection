# Imports
import numpy as np
import pandas as pd
import os
import time

# internet
import urllib
import codecs
import base64

# model deploy
import streamlit as st
import streamlit.components.v1 as stc
import matplotlib
matplotlib.use('Agg') # streamlit needs backend Agg

# visualization
import matplotlib.pyplot as plt
import shap

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
    df_Xtrain,df_Xtest,ser_ytrain,ser_ytest = st_load_data(path_data_raw)
    ytest = np.array(ser_ytest).flatten()

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
        st.text(f"Using data: test_raw.csv")
        st.text(f"Shape of Test Data: {df_Xtest.shape}")
        st.dataframe(df_Xtest.head())

    # Test Data Description
    st_header("Test Data Description")

    # Show Column Names
    if st.checkbox("Show Columns Names"):
        st.write(df_Xtest.columns.tolist())

    # Model Prediction
    st_header("Model Prediction and Evaluation")
    model = joblib.load(path_model_lgb)
    yprobs = model.predict(np.array(df_Xtest)).flatten()
    ypreds = np.array(yprobs>0.5).astype(int)

    return (model,df_Xtest,ytest,ypreds,yprobs)

def model_evaluation(model,df_Xtest,ytest,ypreds,yprobs):
    df_eval = get_binary_classification_scalar_metrics('lightgbm',
                model,df_Xtest,ytest,ypreds,yprobs,
                desc='').round(4)
    st.dataframe(df_eval.T)

    # scikit-plot
    txt = """\
    This is the fraud detection model. Look at the bottom left corner of
    confusion matrix shown below. This number should be small.
    These are False Negatives (FN), which are the true frauds but the model classified them as non-frauds.
    In this fraud detection we aim to reduce the metrics: FN, Recall, and area
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
    fig = plotly_binary_clf_evaluation('lightgbm',model,ytest,ypreds,yprobs)
    st.plotly_chart(fig)

def st_shap(plot, height=None):
    # This needs shap >= 0.36
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    stc.html(shap_html, height=height)

def model_evaluation_shap(model,df_Xtest):
    # derived quantities
    features = list(df_Xtest.columns)

    # shap
    explainer = shap.TreeExplainer(model)
    shap_values = np.array(explainer.shap_values(df_Xtest))
    expected_values = explainer.expected_value

    # make shape of shape_values and expected_value same as df_Xtest
    shap_values = shap_values[1] # values for class 1
    expected_values = expected_values[1]

    assert shap_values.shape == df_Xtest.shape
    assert expected_values.shape == df_Xtest.shape

    ## force plot
    st.subheader("Force plot for first row of test data.")
    idx = st.slider("Select the row number:",0,100)
    plot = shap.force_plot(expected_values,
                    shap_values[idx,:],
                    df_Xtest.iloc[idx,:]
                )
    st_shap(plot)

    st.subheader("Force plot for first N rows of test data.")
    NUM = st.slider("Select N first rows of test data:",1,100)
    plot = shap.force_plot(expected_values,
                    shap_values[:NUM,:],
                    df_Xtest.iloc[:NUM,:]
                )
    st_shap(plot)

    st.subheader("Summary plot of test data")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, df_Xtest)
    st.pyplot(fig)

    st.subheader("Summary plot (barplot) of test data")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, df_Xtest, plot_type='bar')
    st.pyplot(fig)
    plt.gca()

    st.subheader("Dependence plot between two features")
    feature_x = st.selectbox("feature_x", features)
    feature_y = st.selectbox("feature_y", features)
    st.text("You selected feature_x = {} and feature_y = {}".format(
        feature_x, feature_y))
    fig, ax = plt.subplots()
    shap.dependence_plot(ind=feature_x, interaction_index=feature_y,
                    shap_values=shap_values,
                    features=df_Xtest,
                    ax=ax)
    st.pyplot(fig)

    st.subheader("Dependence plot for Nth rank feature")
    st.text("Note: 0 is the most importanct feature not 1.\nThe y-axis feature is selected automatically.")
    rank_n = st.slider("rank_n", 0, len(features)-1,0)
    fig, ax = plt.subplots()
    shap.dependence_plot(ind="rank("+str(rank_n)+")",
                    shap_values=shap_values,
                    features=df_Xtest,
                    ax=ax)
    st.pyplot(fig)

def st_display_html(path_html,width=1000,height=500):
    fh_report = codecs.open(path_html,'r')
    page = fh_report.read()
    stc.html(page,height=height,width=width,scrolling=True)

if __name__ == "__main__":
    menu = ["Home","About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        model,df_Xtest,ytest,ypreds,yprobs= home()
        model_evaluation(model,df_Xtest,ytest,ypreds,yprobs)
        st_header("Model Interpretation using SHAP")
        model_evaluation_shap(model,df_Xtest)

    elif choice == "About":
            st_display_html('about.html',width=600,height=800)
