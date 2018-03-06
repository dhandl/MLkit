import os

from separation import getSeparation
from getRatio import getRatio
# arrays
import numpy as np
import pandas as pd

#scipy
from scipy.stats import ks_2samp

# matplot lib for plotting
import matplotlib
import matplotlib.pyplot as plt

def plot_TrainTest_score(sig_predicted_train, sig_predicted_test, sig_w_train, sig_w_test, bkg_predicted_train, bkg_predicted_test, bkg_w_train, bkg_w_test, binning, fileName="KS_test", normed=False, save=False, ratio=True):
  fig = plt.figure(figsize=(8,6))
  if ratio:
    ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=3)
    #ax1.xaxis.set_ticks([])
  else: 
    ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=4)
  ax1.tick_params(direction='in')
  ax1.set_xlim((binning[1], binning[2]))
  ax1.xaxis.set_ticks_position('both')
  ax1.yaxis.set_ticks_position('both')

  s_histTrain, s_binsTrain, s_patchesTrain = plt.hist(sig_predicted_train.ravel(), weights=sig_w_train, histtype='stepfilled', color='r', label='Signal (Training)', alpha=0.5, bins=binning[0], range=(binning[1], binning[2]), normed=normed)
  b_histTrain, b_binsTrain, b_patchesTrain = plt.hist(bkg_predicted_train.ravel(), weights=bkg_w_train, histtype='stepfilled', color='b', label='Background (Training)', alpha=0.5, bins=binning[0], range=(binning[1], binning[2]), normed=normed)

  s_histTest, s_binsTest = np.histogram(sig_predicted_test.ravel(), weights=sig_w_test, bins=binning[0], range=(binning[1], binning[2]), normed=normed)
  b_histTest, b_binsTest = np.histogram(bkg_predicted_test.ravel(), weights=bkg_w_test, bins=binning[0], range=(binning[1], binning[2]), normed=normed)

  width = (s_binsTrain[1] - s_binsTrain[0])
  center = (s_binsTrain[:-1] + s_binsTrain[1:]) / 2
  plt.errorbar(center, s_histTest, fmt='o', c='r', label='Signal (Testing)') # TODO define yerr = sqrt( sum w^2 ) per bin!
  plt.errorbar(center, b_histTest, fmt='o', c='b', label='Background (Testing)') # TODO define yerr = sqrt( sum w^2 ) per bin!

  ks_sig, ks_sig_p = ks_2samp(s_histTrain, s_histTest)
  ks_bkg, ks_bkg_p = ks_2samp(b_histTrain, b_histTest)
  #sep = getSeparation(s_histTest, s_binsTest, b_histTest, b_binsTest)

  #print sep
  if normed:
    ax1.set(ylabel="a. u.")
  else:
    ax1.set(ylabel="Events")
  leg = plt.legend(loc="best", frameon=False)
  p = leg.get_window_extent()
  #ax.annotate('KS Test S (B): %.3f (%.3f)'%(ks_sig, ks_bkg),(p.p0[0], p.p1[1]), (p.p0[0], p.p1[1]), xycoords='figure pixels', zorder=9)
  ax1.text(0.65, 0.7, "KS Test S (B): %.3f (%.3f)"%(ks_sig, ks_bkg), transform=ax1.transAxes)
  #ax1.text(0.65, 0.70, '$<S^2>$ = %.3f'%(sep), transform=ax1.transAxes)
  #ax.text(0.55, 0.7, "KS p-value S (B): %.3f (%.3f)"%(ks_sig_p, ks_bkg_p), transform=ax.transAxes)

  if ratio:
    ax2 = plt.subplot2grid((4,4), (3,0), colspan=4, rowspan=1)
    getRatio(s_histTest, s_binsTest, b_histTest, b_binsTest)
    ax2.set(xlabel='Output score', ylabel='S/B')
    ax2.set_xlim((binning[1],binning[2]))
    ax2.set_ylim((0,2))
    ax2.grid()
    ax2.tick_params(direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')

  ax1.set(xlabel='Output score')

  if save:
    plt.savefig(fileName+".pdf")
    plt.savefig(fileName+".png")
    plt.close()

