"""

 EXAMPLE
   %reset -f
   %matplotlib auto
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
   from TRANSFORMERS.df2onehot import df2onehot
   import SUPERVISED.twoClassSummary as twoClassSummary


   gb=GradientBoostingClassifier()

   import VIZ.CAP as CAP
   
   ######## Load some data ######
   df=pd.read_csv('../DATA/OTHER/titanic/titanic_train.csv')
   dfc=df2onehot(df)[0]
   dfc.dropna(inplace=True)
   y=dfc['Survived'].astype(float).values
   del dfc['Survived']
   [X_train, X_test, y_train, y_test]=train_test_split(dfc, y, test_size=0.2)

   # Prediction
   model=gb.fit(X_train, y_train)
   P=model.predict_proba(X_test)
   scores=twoClassSummary.allresults(y_test, P[:,1])

   SEE ALSO
   ROCplot, CAP, twoClassSummary
"""

#https://towardsdatascience.com/machine-learning-classifier-evaluation-using-roc-and-cap-curves-7db60fe6b716
#https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
#http://arogozhnikov.github.io/2015/10/05/roc-curve.html

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import  matthews_corrcoef
from sklearn.metrics import average_precision_score, precision_recall_curve
from funcsigs import signature
import SUPERVISED.confmatrix as confmatrix
import VIZ.CAP as CAP
#from VIZ.ROCplot import ROCplot

#%% Main function for all two class results
def allresults(y_true, y_pred_proba, threshold=0.5, title='', classnames=['class1','class2'], showfig=True, verbose=3):
    assert ~isinstance(y_pred_proba, pd.DataFrame), 'pandas DataFrame not allowed as input for y_pred_proba'
    assert ~isinstance(y_true, pd.DataFrame), 'pandas DataFrame not allowed as input for y_true'
    
    if verbose>=3: print('[CLASSIFICATION SUMMARY] Initializing..')
    ax=dict()
    ax[0]=[None,None]
    ax[1]=[None,None]
    # Create classification report
    out=twoClassStats(y_true, y_pred_proba, threshold=threshold, verbose=verbose)
    # Create empty figure
    if showfig: [fig, ax]=plt.subplots(2,2,figsize=(28,16))
    # ROC plot
    _=ROCplot(y_true, y_pred_proba, threshold=threshold, title=title, ax=ax[0][0], showfig=showfig, verbose=0)
    # CAP plot
    out['CAP']=CAP.plot(y_true, y_pred_proba, ax=ax[0][1], showfig=showfig)
    # Probability plot
    out['TPFP']=PROBplot(y_true, y_pred_proba, threshold=threshold, title=title, ax=ax[1][1], showfig=showfig)
    # Probability plot
    PRcurve(y_true, y_pred_proba, title=title, ax=ax[1][0], showfig=showfig)
    # Confusion matrix
    out['confmatrix']=confmatrix.twoclass(y_true, y_pred_proba, threshold=threshold, classnames=classnames, title=title, cmap=plt.cm.Blues, showfig=showfig, verbose=verbose)
    # Show plot
    if showfig: plt.show()
    # Return
    return(out)

#%% Main function for all two class results
def twoClassStats(y_true, y_pred_proba, threshold=0.5, verbose=3):
    # ROC curve
    [fpr, tpr, thresholds] = roc_curve(y_true, y_pred_proba)
    # AUC
    roc_auc = auc(fpr, tpr)
    if verbose>=3: print('[CLASSIFICATION SUMMARY] AUC: %.2f' %(roc_auc))
    # F1 score
    f1score = f1_score(y_true, y_pred_proba>=threshold)
    if verbose>=3: print('[CLASSIFICATION SUMMARY] F1: %.2f' %(f1score))
    # Classification report
    clreport = classification_report(y_true, (y_pred_proba>=threshold).astype(int));
    # Kappa score
    kappscore = cohen_kappa_score(y_true, y_pred_proba>=threshold)
    if verbose>=3: print('[CLASSIFICATION SUMMARY] Kappa: %.2f' %(kappscore))
    # Average precision score
    average_precision = average_precision_score(y_true, y_pred_proba)
    # Recall
    [precision, recall, _] = precision_recall_curve(y_true, y_pred_proba)
    # MCC (Matthews Correlation Coefficient)
    outMCC = MCC(y_true, y_pred_proba, threshold=threshold)

    # Store and return
    out = dict()
    out['auc'] = roc_auc
    out['f1'] = f1score
    out['kappa'] = kappscore
    out['report'] = clreport
    out['thresholds'] = thresholds
    out['fpr'] = fpr
    out['tpr'] = tpr
    out['average_precision']=average_precision
    out['precision']=precision
    out['recall']=recall
    out['MCC']=outMCC
    
    return(out)

#%% MCC (Matthews Correlation Coefficient)
def MCC(y_true, y_pred_proba, threshold=0.5, verbose=3):
    '''
    MCC is extremely good metric for the imbalanced classification and can be 
    safely used even classes are very different in sizes.
    Ranges between [−1,1]
         1: perfect prediction
         0: random prediction
        −1: Total disagreement between predicted scores and true labels values.
    
    '''
    y_true = (y_true).astype(int)
    y_pred = (y_pred_proba>=threshold).astype(int)
    # MCC score
    MCC = matthews_corrcoef(y_true,y_pred)
    
    return(MCC)
     
#%% ROC plot
def ROCplot(y_true, y_pred_proba, threshold=0.5, title='', ax=None, showfig=True, verbose=3):
    # Create classification report
    out=twoClassStats(y_true, y_pred_proba, threshold=threshold, verbose=verbose)

    # Plot figure
    if showfig:
        if isinstance(ax,type(None)):
            [fig,ax]= plt.subplots(figsize=(12,8))
    
        lw = 2
        ax.plot(out['fpr'], out['tpr'], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % out['auc'])
        ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('[%s] Receiver operating characteristic. F1:%.3f, Kappa:%.3f' %(title, out['f1'], out['kappa']))
        ax.legend(loc="lower right")
        ax.grid(True)

    return(out)

#%% Creating probabilty classification plot
def PROBplot(y_true, y_pred_proba, threshold=0.5, title='', ax=None, showfig=True):
    tmpout=pd.DataFrame()
    tmpout['pred_class']=y_pred_proba
    tmpout['true']=y_true
    tmpout.sort_values(by=['true','pred_class'], ascending=False, inplace=True)
    tmpout.reset_index(drop=True, inplace=True)

    Itrue=tmpout['true'].values==1
    # True Positive class
    Itp=(tmpout['pred_class']>=threshold) & (Itrue)
    # True negative class
    Itn=(tmpout['pred_class']<threshold) & (Itrue)
    # False positives
    Ifp=(tmpout['pred_class']>=threshold) & (Itrue==False)
    # False negative class
    Ifn=(tmpout['pred_class']<threshold) & (Itrue==False)

    # Plot figure
    if showfig:
        if isinstance(ax,type(None)): [fig,ax]= plt.subplots(figsize=(20,8))
        # True Positive class
        ax.plot(tmpout['pred_class'].loc[Itp], 'g.',label='True Positive')
        # True negative class
        ax.plot(tmpout['pred_class'].loc[Itn], 'gx',label='True negative')
        # False positives
        ax.plot(tmpout['pred_class'].loc[Ifp], 'rx',label='False positive')
        # False negative class
        ax.plot(tmpout['pred_class'].loc[Ifn], 'r.',label='False negative')
        # Styling
        ax.hlines(threshold, 0,len(Itrue), 'r', linestyles='dashed')
        ax.set_ylim([0,1])
        ax.set_ylabel('P(class | X)')
        ax.set_xlabel('Samples')
        ax.grid(True)
        ax.legend()
        plt.show()
        
#    # Plot figure
#    if showfig:
#        if isinstance(ax,type(None)):
#            [fig,ax]= plt.subplots(figsize=(20,8))
#     
#        Itrue=tmpout['true'].values==1
#        # True Positive class
#        Itp=(tmpout['pred_class']>=threshold) & (Itrue)
#        ax.plot(tmpout['pred_class'].loc[Itp], 'g.',label='True Positive')
#        # True negative class
#        Itn=(tmpout['pred_class']<threshold) & (Itrue)
#        ax.plot(tmpout['pred_class'].loc[Itn], 'gx',label='True negative')
#        # False positives
#        Ifp=(tmpout['pred_class']>=threshold) & (Itrue==False)
#        ax.plot(tmpout['pred_class'].loc[Ifp], 'rx',label='False positive')
#        # False negative class
#        Ifn=(tmpout['pred_class']<threshold) & (Itrue==False)
#        ax.plot(tmpout['pred_class'].loc[Ifn], 'r.',label='False negative')
#        # Styling
#        ax.hlines(threshold, 0,len(Itrue), 'r', linestyles='dashed')
#        ax.set_ylim([0,1])
#        ax.set_ylabel('P(class | X)')
#        ax.set_xlabel('Samples')
#        ax.grid(True)
#        ax.legend()
#        plt.show()

    out=dict()
    out['TP']=np.where(Itp)[0]
    out['TN']=np.where(Itn)[0]
    out['FP']=np.where(Ifp)[0]
    out['FN']=np.where(Ifn)[0]
    
    return(out)

#%% Creating probabilty classification plot
def PRcurve(y_true, y_pred_proba, title='', ax=None, showfig=True):
    '''
    A better metric in an imbalanced situation is the AUC PR (Area Under the Curve Precision Recall), or also called AP (Average Precision).
    If the precision decreases when we increase the recall, it shows that we have to choose a prediction thresold adapted to our needs. 
    If our goal is to have a high recall, we should set a low prediction thresold that will allow us to detect most of the observations of the positive class, but with a low precision. On the contrary, if we want to be really confident about our predictions but don't mind about not finding all the positive observations, we should set a high thresold that will get us a high precision and a low recall.
    In order to know if our model performs better than another classifier, we can simply use the AP metric. To assess the quality of our model, we can compare it to a simple decision baseline. Let's take a random classifier as a baseline here that would predict half of the time 1 and half of the time 0 for the label.
    Such a classifier would have a precision of 4.3%, which corresponds to the proportion of positive observations. For every recall value the precision would stay the same, and this would lead us to an AP of 0.043. The AP of our model is approximately 0.35, which is more than 8 times higher than the AP of the random method. This means that our model has a good predictive power.
    '''
    
    average_precision = average_precision_score(y_true, y_pred_proba)
    [precision, recall, _] = precision_recall_curve(y_true, y_pred_proba)
    
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})

    # Plot figure
    if showfig:
        if isinstance(ax,type(None)):
            [fig,ax]= plt.subplots(figsize=(15,8))
        
        ax.step(recall, precision, color='b', alpha=0.2, where='post')
        ax.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        ax.set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
        ax.grid(True)
    
    out=dict()
    out['average_precision']=average_precision
    out['precision']=precision
    out['recall']=recall
    return(out)