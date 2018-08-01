import os
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import AtlasStyle_mpl

from sklearn.metrics import roc_curve, auc, roc_auc_score
from copy import deepcopy

def plot_ROC(y_train, y_test, y_predict_train, y_predict_test, save=False, fileName='Test'):
    '''
    Plots the ROC-curves for different backgrounds 
    '''
    
    #Binary Classification
    
    print 'Generating ROC-plots for binary Classification...'
    
    y_train_binary = deepcopy(y_train)
    y_train_binary[y_train != 0] = 0.
    y_train_binary[y_train == 0] = 1.
     
    y_test_binary = deepcopy(y_test)
    y_test_binary[y_test != 0] = 0.
    y_test_binary[y_test == 0] = 1.
  
    y_predict_test_binary = y_predict_test[:,0]
    y_predict_train_binary = y_predict_train[:,0]
    
    #print y_test, y_train
    #print np.unique(y_test, return_counts=True)
    #print np.unique(y_train, return_counts=True)
    
    fpr_train_binary, tpr_train_binary, thresholds_train_binary = roc_curve(y_train_binary, y_predict_train_binary)
    fpr_test_binary, tpr_test_binary, thresholds_test_binary = roc_curve(y_test_binary, y_predict_test_binary)
    
    auc_train_binary = roc_auc_score(y_train_binary, y_predict_train_binary)
    auc_test_binary = roc_auc_score(y_test_binary, y_predict_test_binary)
    
    #auc_train = auc(fpr_train, tpr_train, reorder=True)
    #auc_test = auc(fpr_test, tpr_test, reorder=True)
    
    fig1, ax1 = plt.subplots(figsize=(8,6))
    plt.plot(tpr_train_binary, 1.-fpr_train_binary, lw=1, label='Training (AUC = %0.2f)'%(auc_train_binary), color='b')
    plt.plot(tpr_test_binary, 1.-fpr_test_binary, lw=1, label='Testing (AUC = %0.2f)'%(auc_test_binary), color='r')
    #plt.plot(fpr_train_binary, 1.-tpr_train_binary, lw=1, label='Training (AUC = %0.2f)'%(auc_train_binary), color='b')
    #plt.plot(fpr_test_binary, 1.-tpr_test_binary, lw=1, label='Testing (AUC = %0.2f)'%(auc_test_binary), color='r')
    plt.plot([0, 1], [1, 0], '--', color=(0.6, 0.6, 0.6), label='Luck')

    ax1.set_xlim([0., 1.])
    ax1.set_ylim([0., 1.])
    ax1.set_xlabel('Signal Efficiency')
    ax1.set_ylabel('Background Rejection')
    #plt.xlabel('fpr')
    #plt.ylabel('1-tpr')
    #plt.title('Receiver operating characteristic')
    plt.legend(loc="best")
    plt.grid()
    AtlasStyle_mpl.ATLASLabel(ax1, 0.022, 0.925, 'Work in progress')
    #AtlasStyle_mpl.LumiLabel(ax1, 0.022, 0.875, lumi=LUMI*0.001)
    if save:
        if not os.path.exists("./plots/"):
            os.makedirs("./plots/")
            print("Creating folder plots")
        plt.savefig("plots/"+fileName+"_ROCcurveBinary.pdf")
        plt.savefig("plots/"+fileName+"_ROCcurveBinary.png")
        plt.close()
        
    #Multiclass Classification
    
    #print 'Generating ROC-plots for Multiclass Classification...'
        
    #y_train_bkg1 = np.array([x for x in y_train if x in [0,1]])
    #y_train_bkg2 = np.array([x for x in y_train if x in [0,2]])
    #y_train_bkg2[y_train_bkg2==2]=1
    #y_train_bkg3 = np.array([x for x in y_train if x in [0,3]])
    #y_train_bkg3[y_train_bkg3==3]=1
    #y_predict_train_bkg1 = y_predict_train[np.logical_or(y_train==0, y_train==1)][:,0]
    #y_predict_train_bkg2 = y_predict_train[np.logical_or(y_train==0, y_train==2)][:,0]
    #y_predict_train_bkg3 = y_predict_train[np.logical_or(y_train==0, y_train==3)][:,0]
    #y_test_bkg1 = np.array([x for x in y_test if x in [0,1]])
    #y_test_bkg2 = np.array([x for x in y_test if x in [0,2]])
    #y_test_bkg2[y_test_bkg2==2]=1
    #y_test_bkg3 = np.array([x for x in y_test if x in [0,3]])
    #y_test_bkg3[y_test_bkg3==3]=1
    #y_predict_test_bkg1 = y_predict_test[np.logical_or(y_test==0, y_test==1)][:,0]
    #y_predict_test_bkg2 = y_predict_test[np.logical_or(y_test==0, y_test==2)][:,0]
    #y_predict_test_bkg3 = y_predict_test[np.logical_or(y_test==0, y_test==3)][:,0]
    
    
    #try:
    #    fpr_train_bkg1, tpr_train_bkg1, thresholds_train_bkg1 = roc_curve(y_train_bkg1, y_predict_train_bkg1)
    #    fpr_test_bkg1, tpr_test_bkg1, thresholds_test_bkg1 = roc_curve(y_test_bkg1, y_predict_test_bkg1)
        
    #    auc_train_bkg1 = roc_auc_score(y_train_bkg1, y_predict_train_bkg1)
    #    auc_test_bkg1 = roc_auc_score(y_test_bkg1, y_predict_test_bkg1)
        
    #    plt.figure('ROC2')
    #    plt.plot(tpr_train_bkg1, 1.-fpr_train_bkg1, lw=1, label=r'$t\bar{t}$ (AUC = %0.2f)'%(auc_train_bkg1), color='b')
    #    plt.figure('ROC3')
    #    plt.plot(tpr_test_bkg1, 1.-fpr_test_bkg1, lw=1, label=r'$t\bar{t}$ (AUC = %0.2f)'%(auc_test_bkg1), color = 'b')
        
    #except ValueError:
    #    print 'Class 1 has not been identified correctly'

    #try:
    #    fpr_train_bkg2, tpr_train_bkg2, thresholds_train_bkg2 = roc_curve(y_train_bkg2, y_predict_train_bkg2)
    #    fpr_test_bkg2, tpr_test_bkg2, thresholds_test_bkg2 = roc_curve(y_test_bkg2, y_predict_test_bkg2)
        
    #    auc_train_bkg2 = roc_auc_score(y_train_bkg2, y_predict_train_bkg2)
    #    auc_test_bkg2 = roc_auc_score(y_test_bkg2, y_predict_test_bkg2)
        
    #    plt.figure('ROC2')
    #    plt.plot(tpr_train_bkg2, 1.-fpr_train_bkg2, lw=1, label='Single Top (AUC = %0.2f)'%(auc_train_bkg2), color='g')
    #    plt.figure('ROC3')
    #    plt.plot(tpr_test_bkg2, 1.-fpr_test_bkg2, lw=1, label='Single Top (AUC = %0.2f)'%(auc_test_bkg2), color='g')
    
    #except ValueError:
    #    print 'Class 2 has not been identified correctly'
    
    
    #try:
    #    fpr_train_bkg3, tpr_train_bkg3, thresholds_train_bkg3 = roc_curve(y_train_bkg3, y_predict_train_bkg3)
    #    fpr_test_bkg3, tpr_test_bkg3, thresholds_test_bkg3 = roc_curve(y_test_bkg3, y_predict_test_bkg3)
        
    #    auc_train_bkg3 = roc_auc_score(y_train_bkg3, y_predict_train_bkg3)
    #    auc_test_bkg3 = roc_auc_score(y_test_bkg3, y_predict_test_bkg3)
    
    #    plt.figure('ROC2')
    #    plt.plot(tpr_train_bkg3, 1.-fpr_train_bkg3, lw=1, label=r'$W$ + jets (AUC = %0.2f)'%(auc_train_bkg3), color='orange')
    #    plt.figure('ROC3')
    #    plt.plot(tpr_test_bkg3, 1.-fpr_test_bkg3, lw=1, label=r'$W$ + jets (AUC = %0.2f)'%(auc_test_bkg3), color='orange')
    
    #except ValueError:
    #    print 'Class 3 has not been identified correctly'
    
    
    #plt.figure('ROC2')
    #plt.plot([0, 1], [1, 0], '--', color=(0.6, 0.6, 0.6), label='Luck')
    #plt.xlim([0., 1.])
    #plt.ylim([0., 1.])
    #plt.xlabel('Signal Efficiency')
    #plt.ylabel('Background Rejection')
    #plt.title('Receiver operating characteristic')
    #plt.legend(loc="best")
    #plt.grid()
    
    #plt.figure('ROC3')
    #plt.plot([0, 1], [1, 0], '--', color=(0.6, 0.6, 0.6), label='Luck')
    #plt.xlim([0., 1.])
    #plt.ylim([0., 1.])
    #plt.xlabel('Signal Efficiency')
    #plt.ylabel('Background Rejection')
    #plt.title('Receiver operating characteristic')
    #plt.legend(loc="best")
    #plt.grid()
    ##plt.show()
    
    
    #if save:
    #    if not os.path.exists("./plots/"):
    #        os.makedirs("./plots/")
    #        print("Creating folder plots")
    #    plt.figure('ROC1')
    #    plt.savefig("plots/"+fileName+"_ROCcurveBinary.pdf")
    #    plt.savefig("plots/"+fileName+"_ROCcurveBinary.png")
    #    plt.close()
    #    plt.figure('ROC2')
    #    plt.savefig("plots/"+fileName+"_ROCcurveMulticlassTraining.pdf")
    #    plt.savefig("plots/"+fileName+"_ROCcurveMulticlassTraining.png")
    #    plt.close()
    #    plt.figure('ROC3')
    #    plt.savefig("plots/"+fileName+"_ROCcurveMulticlassTesting.pdf")
    #    plt.savefig("plots/"+fileName+"_ROCcurveMulticlassTesting.png")
    #    plt.close()
