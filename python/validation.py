import os, copy, sys
import glob
import random 
import itertools

# ROOT
import ROOT
import ROOT.RooStats as rs

# for arrays
import pandas as pd
import numpy as np
from array import array

# scipy 
from scipy.stats import ks_2samp

# matplot lib for plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# scikit-learn
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc

import seaborn as sns

def plotVar(var, samples, binning, fileName, xTitle, yTitle="Events", legend=None, unit=None, log=False, normed=False, savePlot=False):
  colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
  fig = plt.figure(figsize=(6,6))
  if type(samples) == tuple:
    if (unit == None) or (unit.lower() == 'gev'): 
      hist, bins, patches = plt.hist(samples[0][var], bins=binning[0], range=(binning[1],binning[2]),\
                                     normed=normed, weights=samples[1], cumulative=False, bottom=None, histtype='step',\
                                     align='mid', orientation='vertical', rwidth=None, log=log, color=colors[0],\
                                     label=legend, stacked=False, hold=None, data=None)
    elif unit.lower() == "mev":
      hist, bins, patches = plt.hist(samples[0][var]*0.001, bins=binning[0], range=(binning[1],binning[2]),\
                                     normed=normed, weights=samples[1], cumulative=False, bottom=None, histtype='step',\
                                     align='mid', orientation='vertical', rwidth=None, log=log, color=colors[0],\
                                     label=legend, stacked=False, hold=None, data=None)
  elif type(samples) == list:
    if (unit == None) or (unit.lower() == 'gev'): 
      hist, bins, patches = plt.hist([s[0][var] for s in samples], bins=binning[0], range=(binning[1],binning[2]),\
                                     normed=normed, weights=[s[1] for s in samples], cumulative=False, bottom=None, histtype='step',\
                                     align='mid', orientation='vertical', rwidth=None, log=log, color=[colors[i] for i in range(len(samples))],\
                                     label=legend, stacked=False, hold=None, data=None)
    elif unit.lower() == "mev":
      hist, bins, patches = plt.hist([s[0][var]*0.001 for s in samples], bins=binning[0], range=(binning[1],binning[2]),\
                                     normed=normed, weights=[s[1] for s in samples], cumulative=False, bottom=None, histtype='step',\
                                     align='mid', orientation='vertical', rwidth=None, log=log, color=[colors[i] for i in range(len(samples))],\
                                     label=legend, stacked=False, hold=None, data=None)
  plt.subplots_adjust(left=0.15)
  plt.xlabel(xTitle)
  if normed:
    plt.ylabel("a. u.")
  else:
    plt.ylabel(yTitle)
  plt.legend(legend, loc='best')
  if save:
    plt.savefig(fileName+".pdf")
    plt.savefig(fileName+".png")
    plt.close()

def plotCorrelation(s_train, b_train, s_test, b_test, nvar, fileName, save=False):
  # define color patette
  cmap = sns.color_palette("RdBu_r", 100)
  if not os.path.exists(plotDir):
    os.makedirs(plotDir)
  # Signal Training
  corr_strain, ax = plt.subplots(figsize=(6,6))
  df = pd.DataFrame(data=s_train, columns = nvar)
  corr = df.corr()
  sns.heatmap(corr, cmap=cmap, vmin=-1., vmax=1., square=True, fmt=".2f", annot=True)
  plt.title("Correlation Signal Training") 
  if save:
    plt.savefig(fileSuffix+"_CorrelationMatrix_SignalTraining.pdf")
    plt.savefig(fileSuffix+"_CorrelationMatrix_SignalTraining.png")
    plt.close()
  # Background Training
  corr_btrain, ax = plt.subplots(figsize=(6,6))
  df = pd.DataFrame(data=b_train, columns = nvar)
  corr = df.corr()
  sns.heatmap(corr, cmap=cmap, vmin=-1., vmax=1., square=True, fmt=".2f", annot=True)
  plt.title("Correlation Background Training") 
  if save:
    plt.savefig(fileSuffix+"_CorrelationMatrix_BackgroundTraining.pdf")
    plt.savefig(fileSuffix+"_CorrelationMatrix_BackgroundTraining.png")
    plt.close()
  # Signal Training
  corr_stest, ax = plt.subplots(figsize=(6,6))
  df = pd.DataFrame(data=s_test, columns = nvar)
  corr = df.corr()
  sns.heatmap(corr, cmap=cmap, vmin=-1., vmax=1., square=True, fmt=".2f", annot=True)
  plt.title("Correlation Signal Test") 
  if save:
    plt.savefig(fileSuffix+"_CorrelationMatrix_SignalTest.pdf")
    plt.savefig(fileSuffix+"_CorrelationMatrix_SignalTest.png")
    plt.close()
  # Background Training
  corr_btest, ax = plt.subplots(figsize=(6,6))
  df = pd.DataFrame(data=b_test, columns = nvar)
  corr = df.corr()
  sns.heatmap(corr, cmap=cmap, vmin=-1., vmax=1., square=True, fmt=".2f", annot=True)
  plt.title("Correlation Background Test") 
  if save:
    plt.savefig(fileSuffix+"_CorrelationMatrix_BackgroundTest.pdf")
    plt.savefig(fileSuffix+"_CorrelationMatrix_BackgroundTest.png")
    plt.close()

def plotROC(classifier, X, y, weight, fileName, wp=None, wp_label=None, save=False):
  if 'keras.models.' in str(type(classifier)):
    y_pred = classifier.predict(X)
  elif 'sklearn.ensemble.' in str(type(classifier)):
    y_pred = classifier.decision_function(X)
  if type(weight) is np.ndarray:
    fpr, tpr, thresholds = roc_curve(y, y_pred, sample_weight=weight)
  else:
    fpr, tpr, thresholds = roc_curve(y, y_pred)
  roc_auc = auc(fpr, tpr, reorder=True)
  plt.plot(tpr, 1.-fpr, lw=1, label='ROC (AUC = %0.2f)'%(roc_auc))
  plt.plot([0, 1], [1, 0], '--', color=(0.6, 0.6, 0.6), label='Luck')
  if type(wp) is list and not None:
    plt.plot([wp[0]], [wp[1]], marker='o', markersize=3, color="red", label=wp_label)
  plt.xlim([0., 1.])
  plt.ylim([0., 1.])
  plt.xlabel('Signal efficiency')
  plt.ylabel('Background rejection')
  plt.title('Receiver operating characteristic')
  plt.legend(loc="lower left")
  plt.grid()
  if save:
    plt.savefig(fileName+".pdf")
    plt.savefig(fileName+".png")
    plt.close()

def compareTrainTest(classifier, X_train, y_train, w_train, X_test, y_test, w_test, fileName, wp=None, wp_label=None, save=False):
  if 'keras.models.' in str(type(classifier)):
    y_pred_train = classifier.predict(X_train)
    y_pred_test = classifier.predict(X_test)

  elif 'sklearn.ensemble.' in str(type(classifier)):
    y_pred_train = classifier.decision_function(X_train)
    y_pred_test = classifier.decision_function(X_test)
  if type(w_train) is np.ndarray and type(w_test) is np.ndarray:
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pred_train, sample_weight=w_train)
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_test, sample_weight=w_test)
  else:
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pred_train)
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_test)
  auc_train = auc(fpr_train, tpr_train, reorder=True)
  auc_test = auc(fpr_test, tpr_test, reorder=True)
  plt.plot(tpr_train, 1.-fpr_train, lw=1, label='Training (AUC = %0.2f)'%(auc_train), color='b')
  plt.plot(tpr_test, 1.-fpr_test, lw=1, label='Testing (AUC = %0.2f)'%(auc_test), color='r')
  plt.plot([0, 1], [1, 0], '--', color=(0.6, 0.6, 0.6), label='Luck')
  if type(wp) is list and not None:
    plt.plot([wp[0]], [wp[1]], marker='o', markersize=3, color="red", label=wp_label)
  plt.xlim([0., 1.])
  plt.ylim([0., 1.])
  plt.xlabel('Signal Efficiency')
  plt.ylabel('Background rejection')
  plt.title('Receiver operating characteristic')
  plt.legend(loc="lower left")
  plt.grid()
  if save:
    plt.savefig(fileName+".pdf")
    plt.savefig(fileName+".png")
    plt.close()

def plotOutputScore(classifier, X, y, w, fileName, title, normed=False, save=False):
  if 'keras.models.' in str(type(classifier)):
    sig_pred = classifier.predict(X[y==1.]).ravel() 
    bkg_pred = classifier.predict(X[y==0.]).ravel() 
  elif 'sklearn.ensemble.' in str(type(classifier)):
    sig_pred = classifier.decision_function(X[y==1.]).ravel()
    bkg_pred = classifier.decision_function(X[y==0.]).ravel()
  fig = plt.figure(figsize=(6,6))
  s_hist, s_bins, s_patches = plt.hist(sig_pred, weights=w[y==1.], histtype='stepfilled', color='r', label='Signal', alpha=0.5, bins=100, range=(-1., 1.), normed=normed)
  b_hist, b_bins, b_patches = plt.hist(bkg_pred, weights=w[y==0.], histtype='stepfilled', color='b', label='Background', alpha=0.5, bins=100, range=(-1., 1.), normed=normed)
  plt.xlabel("Output score")
  if normed:
    plt.ylabel("a. u.")
  else:
    plt.ylabel("Events")
  plt.title(title)
  plt.legend(loc='best')   
  if save:
    plt.savefig(fileName+".pdf")
    plt.savefig(fileName+".png")
    plt.close()

def KolmogorovTest(classifier, X_train, y_train, w_train, X_test, y_test, w_test, binning, fileName="KS_test", normed=False, save=False):
  if 'keras.models.' in str(type(classifier)):
    sig_predicted_train = classifier.predict(X_train[y_train==1.]).ravel()
    sig_predicted_test = classifier.predict(X_test[y_test==1.]).ravel()
    bkg_predicted_train = classifier.predict(X_train[y_train==0.]).ravel()
    bkg_predicted_test = classifier.predict(X_test[y_test==0.]).ravel()
  elif 'sklearn.ensemble.' in str(type(classifier)):
    sig_predicted_train = classifier.decision_function(X_train[y_train==1.]).ravel()
    sig_predicted_test = classifier.decision_function(X_test[y_test==1.]).ravel()
    bkg_predicted_train = classifier.decision_function(X_train[y_train==0.]).ravel()
    bkg_predicted_test = classifier.decision_function(X_test[y_test==0.]).ravel()
  fig = plt.figure(figsize=(6,6))
  s_histTrain, s_binsTrain, s_patchesTrain = plt.hist(sig_predicted_train.ravel(), weights=w_train[y_train==1.], histtype='stepfilled', color='r', label='Signal (Training)', alpha=0.5, bins=binning[0], range=(binning[1], binning[2]), normed=normed)
  b_histTrain, b_binsTrain, b_patchesTrain = plt.hist(bkg_predicted_train.ravel(), weights=w_train[y_train==0.], histtype='stepfilled', color='b', label='Background (Training)', alpha=0.5, bins=binning[0], range=(binning[1], binning[2]), normed=normed)
  s_histTest, s_binsTest = np.histogram(sig_predicted_test.ravel(), weights=w_test[y_test==1.], bins=binning[0], range=(binning[1], binning[2]), normed=normed)
  b_histTest, b_binsTest = np.histogram(bkg_predicted_test.ravel(), weights=w_test[y_test==0.], bins=binning[0], range=(binning[1], binning[2]), normed=normed)
  width = (s_binsTrain[1] - s_binsTrain[0])
  center = (s_binsTrain[:-1] + s_binsTrain[1:]) / 2
  plt.errorbar(center, s_histTest, fmt='o', c='r', label='Signal (Testing)') # TODO define yerr = sqrt( sum w^2 ) per bin!
  plt.errorbar(center, b_histTest, fmt='o', c='b', label='Background (Testing)') # TODO define yerr = sqrt( sum w^2 ) per bin!
  ks_sig, ks_sig_p = ks_2samp(s_histTrain, s_histTest)
  ks_bkg, ks_bkg_p = ks_2samp(b_histTrain, b_histTest)
  plt.xlabel("Output score")
  if normed:
    plt.ylabel("a. u.")
  else:
    plt.ylabel("Events")
  plt.legend(loc='best')
  ax = fig.add_subplot(111)
  ax.text(0.55, 0.75, "KS Test S (B): %.3f (%.3f)"%(ks_sig, ks_bkg), transform=ax.transAxes)
  ax.text(0.55, 0.7, "KS p-value S (B): %.3f (%.3f)"%(ks_sig_p, ks_bkg_p), transform=ax.transAxes)
  if save:
    plt.savefig(fileName+".pdf")
    plt.savefig(fileName+".png")
    plt.close()

def plotSignificance(classifier, X, y, w, score=(-1., 1.), fileName="Significance", save=False):
  if 'keras.models.' in str(type(classifier)):
    sig_predicted = classifier.predict(X[y==1.]).ravel()
    bkg_predicted = classifier.predict(X[y==0.]).ravel()
  elif 'sklearn.ensemble.' in str(type(classifier)):
    sig_predicted = classifier.decision_function(X[y==1.]).ravel()
    bkg_predicted = classifier.decision_function(X[y==0.]).ravel()
  fig = plt.figure(figsize=(6, 6))
  ax1=fig.add_subplot(111, label="significance")
  ax2=fig.add_subplot(111, label="sig", frame_on=False)
  ax3=fig.add_subplot(111, label="bkg", frame_on=False)
  bins = abs(score[1]-(score[0]))*100
  # create an array and initialize it with -1.
  Z = np.ones(int(bins))
  Z = np.negative(Z)
  x = np.zeros(int(bins))
  sumW_sig = np.zeros(int(bins))
  sumW_bkg = np.zeros(int(bins))
  # split weights in signal and background
  w_sig = w[y==1.]
  nEventsSig = w_sig.sum() 
  w_bkg = w[y==0.]
  nEventsBkg = w_bkg.sum()
  di = 0.01
  for i in range(int(bins)):
    x[i] = score[0] + i * di
    S = w_sig[sig_predicted>=x[i]].sum() # sum of weights
    dS = sum(w_sig[sig_predicted>=x[i]]**2) # sum of squared weights
    B = w_bkg[bkg_predicted>=x[i]].sum()
    dB = sum(w_bkg[bkg_predicted>=x[i]]**2)
    tot_rel = np.sqrt(0.25*0.25 + (dB/B)*(dB/B) )
    Z[i] = rs.NumberCountingUtils.BinomialExpZ(S, B, dB)
    sumW_sig[i] = S/nEventsSig
    sumW_bkg[i] = B/nEventsBkg
  ax1.plot(x, Z, 'k')
  ax1.axhline(y=3, c='lightcoral')
  ax1.axhline(y=5, c='lightcoral')
  ax1.set_xlabel('Cut in x')
  ax1.set_ylabel('Zn')
  ax2.plot(x, sumW_sig, 'r-')
  ax2.get_xaxis().set_visible(False)
  ax2.yaxis.tick_right()
  ax2.set_ylabel('cum. fraction', color="royalblue")       
  ax2.yaxis.set_label_position('right') 
  ax2.spines['right'].set_color('royalblue')
  ax2.tick_params(axis='y', color="royalblue")
  ax3.plot(x, sumW_bkg, 'b--')
  ax3.get_xaxis().set_visible(False)
  ax3.get_yaxis().set_visible(False)
  plt.grid()
  plt.title('Significance')
  plt.legend(loc='best')
  if save:
    plt.savefig(fileName+'.pdf')
    plt.savefig(fileName+'.png')
    plt.close()

def plotSigVersusEff(classifier, X, y, w, wp, wp_label, label='ML classifier', fileName="Significance", save=False):
  if 'keras.models.' in str(type(classifier)):
    sig_predicted = classifier.predict(X[y==1.]).ravel()
    bkg_predicted = classifier.predict(X[y==0.]).ravel()
  elif 'sklearn.ensemble.' in str(type(classifier)):
    sig_predicted = classifier.decision_function(X[y==1.]).ravel()
    bkg_predicted = classifier.decision_function(X[y==0.]).ravel()
  fig = plt.figure(figsize=(6, 6))
  ax1=fig.add_subplot(111, label="significance")
  bins = 100
  # create an array and initialize it with -1.
  Z = np.ones(int(bins))
  Z = np.negative(Z)
  x = np.zeros(int(bins))
  sumW_sig = np.zeros(int(bins))
  sumW_bkg = np.zeros(int(bins))
  # split weights in signal and background
  w_sig = w[y==1.]
  nEventsSig = w_sig.sum() 
  w_bkg = w[y==0.]
  nEventsBkg = w_bkg.sum()
  di = 0.01
  for i in range(bins):
    x[i] = i * di
    S = w_sig[sig_predicted>=x[i]].sum() # sum of weights
    dS = sum(w_sig[sig_predicted>=x[i]]**2) # sum of squared weights
    B = w_bkg[bkg_predicted>=x[i]].sum()
    dB = sum(w_bkg[bkg_predicted>=x[i]]**2)
    tot_rel = np.sqrt(0.25*0.25 + (dB/B)*(dB/B) )
    Z[i] = rs.NumberCountingUtils.BinomialExpZ(S, B, dB)
    sumW_sig[i] = S/nEventsSig
    sumW_bkg[i] = B/nEventsBkg
  ax1.plot(sumW_sig, Z, 'k', label=label)
  ax1.axhline(y=3, c='lightcoral')
  ax1.axhline(y=5, c='lightcoral')
  ax1.set_xlabel('Signal efficiency')
  ax1.set_ylabel('Zn')
  if type(wp) is list and not None:
    plt.plot([wp[0]], [wp[1]], marker='o', markersize=3, color="red", label=wp_label)
  plt.grid()
  plt.title('Significance')
  plt.legend(loc='best')
  if save:
    plt.savefig(fileName+'.pdf')
    plt.savefig(fileName+'.png')
    plt.close()

def getConfusioMatrix(y_test, y_predict, w_test, precision=2):
  yhat_cls = y_predict.round()
  cnf_matrix = confusion_matrix(y_test, yhat_cls, sample_weight=w_test)
  np.set_printoptions(precision=2)
  return cnf_matrix

def plot_confusion_matrix(classifier, X, y, w, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
  if 'keras.models.' in str(type(classifier)):
    y_predict = classifier.predict(X)
  elif 'sklearn.ensemble.' in str(type(classifier)):
    y_predict = classifier.decision_function(X)
  cm = getConfusionMatrix(y, y_predict, w)

  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')
  
  print(cm)
  
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)
  
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")
  
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

