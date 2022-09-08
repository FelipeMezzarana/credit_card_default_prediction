"""
Calculates metrics and shows relevant charts for classification models
@author: Felipe S. Mezzarana
"""
 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score

class Metrics:
    
    def __init__(self, y_true,y_pred):
        
        """Create the Metrics object and define the desired metrics as attributes.

        Keyword arguments:
        y_true -- list of predict default
        y_pred -- list of actual default
        """
        
        self.y_pred = y_pred
        self.y_true = y_true
        
        # Recall --  Tp / (Tp + Fn) or (True positive/Total Actual Positive)
        self.recall = recall_score(y_true, y_pred)
        # Precision -- Tp/(Tp + Fp) or (True positive/Total Predicted Positive)
        self.precision = precision_score(y_true, y_pred) 
        # F1 -- 2*precision*recall /(precision + recall)
        self.f1 = 2*self.precision*self.recall/(self.precision+self.recall)
        # AUC - area under roc curve
        self.auc = roc_auc_score(y_true, y_pred)
        # Values of: tp,tn,fp,fn
        self.conf_matrix = confusion_matrix(self.y_true, self.y_pred)
        
    def show_metrics(self):
        
        """Print metrics"""
        
        print(f'Métrics:\n\nRecall: {round(self.recall,3)}'
              f'\nPrecision: {round(self.precision,3)}'
              f'\nF1: {round(self.f1,3)}'
              f'\nAUC: {round(self.auc,3)}')
        
    def show_charts(self, chart = 'all'):
        
        """plot metrics charts
        Charts may be: Confusion Matrix, ROC Curve or both
        
        Keyword arguments:
        chart -- string to define what metric to plot. Can be: 'roc','conf_matrix' or 'all' (default 'all')
        """
        
        if chart == 'all':
            fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))
        elif chart == 'conf_matrix':
            fig, ax1 = plt.subplots(figsize=(5,5))
        elif chart == 'roc':
            fig, ax2 = plt.subplots(figsize=(9,5))
        else:
            return print("Please choose a valid chart value: 'conf_matrix', 'roc' or 'all'")
        
        if chart == 'all' or chart == 'conf_matrix':
            # Confusion Matrix plot
            ConfusionMatrixDisplay(self.conf_matrix).plot(ax =ax1)
            ax1.set_title("Confusion Matrix",fontsize = 18)
            ax1.set_xlabel("Predicted Default",fontsize=14)
            ax1.set_ylabel("Actual Default",fontsize=14)
            ax1.tick_params(labelsize=12)
            
        if chart == 'all' or chart == 'roc':
            # ROC curve plot
            RocCurveDisplay.from_predictions(self.y_true,self.y_pred,ax = ax2)
            ax2.set_title("ROC Curve",fontsize = 18, loc = 'left')
            ax2.set_xlabel("False Positive Rate (Specificity)",fontsize=12,loc = 'left')
            ax2.set_ylabel("True Positive Rate (Recall)",fontsize=12,loc = 'top')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.tick_params(labelsize=12)
            
        plt.show()
   
    
    def show_all(self):
        
        """Plot metrics chart and print metrics values.
        *Confusion Matrix
        *ROC Curve
        *Recall
        *Precision
        *F1
        """
        
        self.show_charts()
        self.show_metrics()
        
        
"""
To prevent overfitting, we will use Stratified K-Folds cross-validator to evaluate our models. 
Therefore, it will be interesting to have a function that receive multiple metrics objects (one for each fold),
generating overlapped charts, mean metrics, and other useful infos. 
"""

def show_mean_metrics(metrics_obj_list):

    """Show mean metrics for multiple tests (K-Folds)
    for n folds:
    Plot overlapped ROC Curve
    Plot sum of Confusions Matrix
    print defined metrics (mean) and respective standard deviations

    Keyword arguments:
    metrics_obj_list -- list of object from metrics class
    """
    
    # Empty arrays to sum Confusion Matrix values from all folds
    conf_matrix_s1 = np.array([0,0])
    conf_matrix_s2 = np.array([0,0])       

    # mean_fpr and tprs will be used to define the mean ROC curve
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    
    # Lists to append individual metrics of each fold
    recall_list,precision_list,f1_list,auc_list = [[]for i in range(4)]
    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(17,6))
    for metrics_obj, i in zip(metrics_obj_list,range(len(metrics_obj_list))):
        
        # Appending metrics values
        recall_list.append(metrics_obj.recall)
        precision_list.append(metrics_obj.precision)
        f1_list.append(metrics_obj.f1)
        auc_list.append(metrics_obj.auc)
        
        # sum Confusion Matrix values
        conf_matrix_s1 = conf_matrix_s1 + metrics_obj.conf_matrix[0]
        conf_matrix_s2 = conf_matrix_s2 + metrics_obj.conf_matrix[1]
        
        # Creating a overlapped ROC curve
        viz = RocCurveDisplay.from_predictions(metrics_obj.y_true,
                                               metrics_obj.y_pred,
                                               name=f"ROC fold {i+1}",lw = 1,alpha=0.4,color = '#72abed',
                                               ax = ax2)
        
        # creating 100 interpelated points between True positive rates and true negative rates 
        # This is the best way to calculate the mean roc curve, 
        # for more info, please visit https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
   
    # Mean points for all folds
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    ax2.plot(mean_fpr,mean_tpr,color="#2349de",
             label =f"Mean ROC (mean AUC = {round(np.mean(auc_list),2)})",
             lw=2, alpha=0.8)
    
    
    # Defining variance around mean ROC Curve to build confidence intervals
    std_tpr = np.std(tprs, axis=0) # Std value for each interpolated point
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1) # Point + std
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0) # Point - std
    ax2.fill_between(mean_fpr,tprs_lower,tprs_upper,color="grey",alpha=0.1,label=r"$\pm$ 1 std. dev.") # Plot interval
    
    
    # Ploting final Confusion Matrix -- sum of all Confusion Matrix of each fold
    final_conf_matrix = np.array([conf_matrix_s1,conf_matrix_s2])
    ConfusionMatrixDisplay(final_conf_matrix).plot(ax =ax1)
    
    # Charts parameters
    # Confusion Matrix
    ax1.set_title("Confusion Matrix (Sum values for all folds)",fontsize = 18)
    ax1.set_xlabel("Predicted Default",fontsize=14)
    ax1.set_ylabel("Actual Default",fontsize=14)
    ax1.tick_params(labelsize=12)
    # ROC Curve
    ax2.set_title("ROC Curve",fontsize = 18,loc = 'left')
    ax2.set_xlabel("False Positive Rate (Specificity)",fontsize=12,loc = 'left')
    ax2.set_ylabel("True Positive Rate (Recall)",fontsize=12,loc = 'top')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(labelsize=12)
    
    plt.legend()
    plt.show()
    
    # Print mean metrics
    print(f'Métrics:\n\nMean Recall = {round(np.mean(recall_list),3)} | Recall Std = {round(np.std(recall_list),2)}'
          f'\nMean Precision = {round(np.mean(precision_list),3)} | Precision Std = {round(np.std(precision_list),2)}'
          f'\nMean F1 = {round(np.mean(f1_list),3)} | F1 Std = {round(np.std(f1_list),2)}'
          f'\nMean AUC = {round(np.mean(auc_list),3)} | AUC Std = {round(np.std(auc_list),2)}')
    