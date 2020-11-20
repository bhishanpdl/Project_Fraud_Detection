__author__ = 'Bhishan Poudel'

__doc__ = """
This module contains various model evaluation utilities.

- get_binary_classification_scalar_metrics

"""

__all__ = ["get_binary_classification_scalar_metrics",
        "plotly_binary_clf_evaluation"
        ]

import numpy as np
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)
import matplotlib
import matplotlib.pyplot as plt
import os
import time

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve,precision_recall_curve

def get_binary_classification_scalar_metrics(model_name,model,
    Xtest,ytest,
    ypreds,
    yprobs,
    desc='',
    df_eval=None,
    style_col="Recall",
    round_=None):
    """Get some scalar metrics for binary classification.

    Parameters
    -----------
    model_name: string
        Name of the model
    model: object
        sklearn model instance
    Xtest: np.array
        Test predictor variable
    ytest: np.array
        Test response variable
    ypreds: np.array
        Test response predictions eg. 0,1
    yprobs: np.array
        Test response probabilities e.g 0.2,0.5,0.9
    df_eval: pandas.DataFrame
        Pandas dataframe for model evaluation.
    style_col: string
        Name of the evaluation output column to style.
    round: int
        Rounding number to round off the displayed data.

.. code-block:: python

    '''

    This is a helper function intended to be used in Jupyter notebooks.

    This function displays various scalar binary classification metrics.
    AA===================================================================

    Accuracy = (TP+TN) / (TP+TN+FP+FN)   Overall accuracy of the model.
    Precision = TP / (TP+FP)  Precision has all P's.
    Useful for email spam detection.
    Recall = TP / (TP+FN) FP is replaced by FN.
    Useful for fraud detection, patient detection.
    F1 = 2 / (1/precision + 1/recall) Harmonic mean of precision and recall.
    Useful when both precision and recall are important.
    AA===================================================================

    Matthews Correlation Coefficient
    https://www.wikiwand.com/en/Matthews_correlation_coefficient
    NOTE: F1-score depends on which class is defined as the positive class.
    F1-score will be high if majority of the classes defined are positive
    or vice versa.

    Matthews Correlation Coefficient is independent of classes frequencies.
    It is given by:

    MCC =     (TP * TN - FP*FN)
        ------------------------------------------------
        sqrt( (TP+FP)   (TP+FN)   (TN+FP)   (TN+FN)  )
    AA===================================================================

    Cohen's Kappa
    https://psychology.wikia.org/wiki/Cohen%27s_kappa
    Cohen's kappa measures the agreement between two raters who each classify
    N items into C mutually exclusive categories.

    Cohen's Kappa is given by:
    k = Pr(a) - Pr(e)
        --------------------
        1 - Pr(e)

    where Pr(a) is the relative observed agreement among raters,
    and Pr(e) is the hypothetical probability of chance agreement,
    using the observed data to calculate the probabilities
    of each observer randomly
    saying each category.
    If the raters are in complete agreement then κ = 1.
    If there is no agreement among the raters
    (other than what would be expected by chance)
    then κ ≤ 0.
    AA===================================================================


    By default the output result is sorted based on Recall.
    Usually Recall is the metric we are inteerested such as
    in case of classification Fraud Detection Modellings.

    In case of spam email detection, we are interested
    in "Precision" of the Spam detection and
    we can choose style_col as 'Precision'.

    In the classification cases such as cat and dogs clasification,
    we may be equally interested in both precision and recall and
    we can choose 'F1' as the style column.

    Example:
    ------------
    df_eval = get_binary_classification_scalar_metrics1(
        model_name_lr,
        clf_lr,
        Xtest,ytest,
        ypreds_lr,
        desc="", df_eval=None) # or, df_eval = df_eval

    '''

    """

    # imports
    from sklearn.metrics import (accuracy_score,precision_score,
                                recall_score,f1_score,matthews_corrcoef,
                                average_precision_score,roc_auc_score,
                                cohen_kappa_score)
    if  not isinstance(df_eval, pd.DataFrame):
        df_eval = pd.DataFrame({'Model': [],
                                'Description':[],
                                'Accuracy':[],
                                'Precision':[],
                                'Recall':[],
                                'F1':[],
                                'Mathews_Correlation_Coefficient': [],
                                'Cohens_Kappa': [],
                                'Area_Under_Precision_Curve': [],
                                'Area_Under_ROC_Curve': [],
                            })

    # scalar metrics
    acc = accuracy_score(ytest,ypreds)
    precision = precision_score(ytest,ypreds)
    recall = recall_score(ytest,ypreds)
    f1 = f1_score(ytest,ypreds)
    mcc = matthews_corrcoef(ytest,ypreds)
    kappa = cohen_kappa_score(ytest, ypreds)

    auprc = average_precision_score(ytest,yprobs)
    auroc = roc_auc_score(ytest, yprobs)

    row = [model_name,desc,acc,precision,recall,f1,mcc,kappa,auprc,auroc]

    df_eval.loc[len(df_eval)] =  row
    df_eval = df_eval.drop_duplicates()\
                .sort_values(style_col,ascending=False)
    df_eval.index = range(len(df_eval))

    df_style = (df_eval.style.apply(lambda ser:
                ['background: lightblue'
                if ser.name ==  style_col
                else '' for _ in ser])
                )
    caption = model_name + ' (' + desc + ')' if desc else model_name
    df_style.set_caption(caption)

    # rounding
    fmt = '{:.' + str(round_) + 'f}'
    if round_ is not None:
        df_style = df_style.format({'Accuracy': fmt })
        df_style = df_style.format({'Precision': fmt })
        df_style = df_style.format({'Recall': fmt })
        df_style = df_style.format({'F1': fmt })
        df_style = df_style.format({'Mathews_Correlation_Coefficient': fmt })
        df_style = df_style.format({'Cohens_Kappa': fmt })

    return df_eval

def plotly_binary_clf_evaluation(model_name,model,
    ytest,ypreds,yprobs,
    ofile=None,auto_open=False) :
    """Plot the binary classification model evaluation.

    Parameters
    -----------
    model_name: str
        Name of the model.
    model: object
        Fitted model. eg. RandomForestClassifier
    ytest: np.array
        Test array.
    ypreds: np.array
        Prediction array
    yprobs: np.array
        Probability array.
    ofile: str
        Name of the output file.
    auto_open: bool
        Whether or not to automatically open the ouput html file.

    Examples
    ---------
    .. code-block:: python
        bp.plotly_binary_clf_evaluation(model_name,model,
    ytest,ypreds,y_score)

    Ref: https://www.kaggle.com/vincentlugat/lightgbm-plotly
    """
    import plotly
    from sklearn.metrics import confusion_matrix
    import plotly.offline as py
    py.init_notebook_mode(connected=True)
    import plotly.graph_objs as go
    import plotly.tools as tls
    import plotly.figure_factory as ff

    #Conf matrix
    conf_matrix = confusion_matrix(ytest, ypreds)
    trace1 = go.Heatmap(z = conf_matrix  ,x = ["0 (Predicted)","1 (Predicted)"],
                        y = ["0 (True)","1 (True)"],xgap = 2, ygap = 2,
                        colorscale = 'Viridis', showscale  = False)

    #Show metrics
    tp = conf_matrix[1,1]
    fn = conf_matrix[1,0]
    fp = conf_matrix[0,1]
    tn = conf_matrix[0,0]
    Accuracy  =  ((tp+tn)/(tp+tn+fp+fn))
    Precision =  (tp/(tp+fp))
    Recall    =  (tp/(tp+fn))
    F1_score  =  (2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))))

    show_metrics = pd.DataFrame(data=[[Accuracy , Precision, Recall, F1_score]],
                                columns=['Accuracy', 'Precision', 'Recall', 'F1_score'])
    show_metrics = show_metrics.T.sort_values(by=0,ascending = False)

    y = show_metrics.index.values.tolist()

    colors = ['gold', 'lightgreen', 'lightcoral', 'lightskyblue']
    trace2 = go.Bar(x = (show_metrics[0].values),
                y = y,
                text = np.round_(show_metrics[0].values,4),
                textposition = 'auto',
                orientation = 'h', opacity = 0.8,marker=dict(
            color=colors,
            line=dict(color='#000000',width=1.5)))

    #Roc curve
    model_roc_auc = round(roc_auc_score(ytest, yprobs) , 3)
    fpr, tpr, t = roc_curve(ytest, yprobs)
    trace3 = go.Scatter(x = fpr,y = tpr,
                        name = "Roc : " + str(model_roc_auc),
                        line = dict(color = ('rgb(22, 96, 167)'),width = 2),
                        fill='tozeroy')
    trace4 = go.Scatter(x = [0,1],y = [0,1],
                        line = dict(color = ('black'),width = 1.5,
                        dash = 'dot'))

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(ytest, yprobs)
    trace5 = go.Scatter(x = recall, y = precision,
                        name = "Precision" + str(precision),
                        line = dict(color = ('lightcoral'),width = 2),
                        fill='tozeroy')

    #Cumulative gain
    pos = pd.get_dummies(ytest).values
    pos = pos[:,1]
    npos = np.sum(pos)
    index = np.argsort(yprobs)
    index = index[::-1]
    sort_pos = pos[index]
    #cumulative sum
    cpos = np.cumsum(sort_pos)
    #recall
    recall = cpos/npos
    #size obs test
    n = ytest.shape[0]
    size = np.arange(start=1,stop=369,step=1)
    #proportion
    size = size / n

    #plots
    trace6 = go.Scatter(x = size,y = recall,
                        name = "Lift curve",
                        line = dict(color = ('gold'),width = 2), fill='tozeroy')

    # subplot titles
    subplot_titles = ('Confusion Matrix',
                    'Metrics',
                    'ROC Curve'+" "+ '('+ str(model_roc_auc)+')',
                    'Precision - Recall Curve',
                    'Cumulative Gains Curve'
                    )
    rows = 3
    specs = [[{}, {}],
            [{}, {}],
            [{'colspan': 2}, None]
            ]

    #Subplots
    try:
        fig = plotly.subplots.make_subplots(rows=rows, cols=2, print_grid=False,
                        specs=specs,
                        subplot_titles=subplot_titles)
    except:
        fig = plotly.tools.make_subplots(rows=rows, cols=2, print_grid=False,
                        specs=specs,
                        subplot_titles=subplot_titles)

    fig.append_trace(trace1,1,1)
    fig.append_trace(trace2,1,2)
    fig.append_trace(trace3,2,1)
    fig.append_trace(trace4,2,1)
    fig.append_trace(trace5,2,2)
    fig.append_trace(trace6,3,1)

    fig['layout'].update(showlegend = False, title = '<b>Model Performance Report</b><br>('+str(model_name)+')',
                        autosize = False, height = 1500,width = 830,
                        plot_bgcolor = 'rgba(240,240,240, 0.95)',
                        paper_bgcolor = 'rgba(240,240,240, 0.95)',
                        margin = dict(b = 195))
    fig["layout"]["xaxis2"].update((dict(range=[0, 1])))
    fig["layout"]["xaxis3"].update(dict(title = "False Positive Rate"))
    fig["layout"]["yaxis3"].update(dict(title = "True Positive Rate"))
    fig["layout"]["xaxis4"].update(dict(title = "Recall"), range = [0,1.05])
    fig["layout"]["yaxis4"].update(dict(title = "Precision"), range = [0,1.05])
    fig["layout"]["xaxis5"].update(dict(title = "Percentage Contacted"))
    fig["layout"]["yaxis5"].update(dict(title = "Percentage Positive Targeted"))
    fig.layout.titlefont.size = 14

    if ofile:
        py.plot(fig, filename=ofile,auto_open=auto_open)

    return fig