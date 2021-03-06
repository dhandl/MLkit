import os

from separation import getSeparation
from getRatio import getRatio
from getSumW2 import getSumW2
# arrays
import numpy as np
import pandas as pd

#scipy
from scipy.stats import ks_2samp

# matplot lib for plotting
import matplotlib
import matplotlib.pyplot as plt

import AtlasStyle_mpl

#dummy object for KS Test in legend
import matplotlib.patches as mpatches

def plot_TrainTest_score(sig_predicted_train, sig_predicted_test, sig_w_train, sig_w_test, bkg_predicted_train, bkg_predicted_test, bkg_w_train, bkg_w_test, binning, fileName='Test', normed=False, save=False, ratio=True, addStr=''):
  print('Plotting the train/test score...')
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

  #s_histTrain, s_binsTrain, s_patchesTrain = plt.hist(sig_predicted_train.ravel(), weights=sig_w_train, histtype='stepfilled', color='r', label='Signal (Training)', alpha=0.5, bins=binning[0], range=(binning[1], binning[2]), density=normed)
  s_histTrain, s_binsTrain, s_patchesTrain = plt.hist(sig_predicted_train.ravel(), weights=None, histtype='stepfilled', color='r', label='Signal (Training)', alpha=0.5, bins=binning[0], range=(binning[1], binning[2]), density=normed)
  #b_histTrain, b_binsTrain, b_patchesTrain = plt.hist(bkg_predicted_train.ravel(), weights=bkg_w_train, histtype='stepfilled', color='b', label='Background (Training)', alpha=0.5, bins=binning[0], range=(binning[1], binning[2]), density=normed)
  b_histTrain, b_binsTrain, b_patchesTrain = plt.hist(bkg_predicted_train.ravel(), weights=None, histtype='stepfilled', color='b', label='Background (Training)', alpha=0.5, bins=binning[0], range=(binning[1], binning[2]), density=normed)

  #s_histTest, s_binsTest = np.histogram(sig_predicted_test.ravel(), weights=sig_w_test, bins=binning[0], range=(binning[1], binning[2]), density=normed)
  s_histTest, s_binsTest = np.histogram(sig_predicted_test.ravel(), weights=None, bins=binning[0], range=(binning[1], binning[2]), density=normed)
  #b_histTest, b_binsTest = np.histogram(bkg_predicted_test.ravel(), weights=bkg_w_test, bins=binning[0], range=(binning[1], binning[2]), density=normed)
  b_histTest, b_binsTest = np.histogram(bkg_predicted_test.ravel(), weights=None, bins=binning[0], range=(binning[1], binning[2]), density=normed)

  width = (s_binsTrain[1] - s_binsTrain[0])
  center = (s_binsTrain[:-1] + s_binsTrain[1:]) / 2
  s_error = plt.errorbar(center, s_histTest, fmt='o', c='r', label='Signal (Testing)') # TODO define yerr = sqrt( sum w^2 ) per bin!
  b_error = plt.errorbar(center, b_histTest, fmt='o', c='b', label='Background (Testing)') # TODO define yerr = sqrt( sum w^2 ) per bin!

  ks_sig, ks_sig_p = ks_2samp(s_histTrain, s_histTest)
  ks_bkg, ks_bkg_p = ks_2samp(b_histTrain, b_histTest)
  #sep = getSeparation(s_histTest, s_binsTest, b_histTest, b_binsTest)
  
  if normed:
    s_w_test = getSumW2(sig_predicted_test.ravel(), sig_w_test, binning)/np.sum(sig_w_test)
    b_w_test = getSumW2(bkg_predicted_test.ravel(), bkg_w_test, binning)/np.sum(bkg_w_test)
  else:
    s_w_test = getSumW2(sig_predicted_test.ravel(), sig_w_test, binning)
    b_w_test = getSumW2(bkg_predicted_test.ravel(), bkg_w_test, binning)
  
  #Proxy artist for KS Test
  
  ks_patch = mpatches.Patch(color='None', label='KS Test S (B): %.3f (%.3f)'%(ks_sig, ks_bkg))

  #print sep
  if normed:
    ax1.set_ylabel('a. u.', horizontalalignment='right', y=1.0)
  else:
    ax1.set_ylabel('Events', horizontalalignment='right', y=1.0)
  leg = plt.legend(loc='best', frameon=False, handles=[s_patchesTrain[0], b_patchesTrain[0], s_error, b_error, ks_patch])
  p = leg.get_window_extent()
  
  #ax.annotate('KS Test S (B): %.3f (%.3f)'%(ks_sig, ks_bkg),(p.p0[0], p.p1[1]), (p.p0[0], p.p1[1]), xycoords='figure pixels', zorder=9)
  #ax1.text(0.65, 0.66, 'KS Test S (B): %.3f (%.3f)'%(ks_sig, ks_bkg), transform=ax1.transAxes) #Former y=0.7
  #ax1.text(0.65, 0.70, '$<S^2>$ = %.3f'%(sep), transform=ax1.transAxes)
  #ax.text(0.55, 0.7, 'KS p-value S (B): %.3f (%.3f)'%(ks_sig_p, ks_bkg_p), transform=ax.transAxes)

  if ratio:
    ax2 = plt.subplot2grid((4,4), (3,0), colspan=4, rowspan=1)
    getRatio(s_histTest, s_binsTest, s_w_test, b_histTest, b_binsTest, b_w_test, 'r')
    ax2.set_xlabel('EPD', horizontalalignment='right', x=1.0) 
    ax2.set_ylabel('S/B')
    ax2.set_xlim((binning[1],binning[2]))
    ax2.set_ylim((0,2))
    ax2.grid()
    ax2.tick_params(direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')

  ax1.set_ylim(0., 1.5*np.maximum(s_histTest.max(), b_histTest.max()))
  ax1.set_xlabel('EPD', horizontalalignment='right', x=1.0)
  AtlasStyle_mpl.ATLASLabel(ax1, 0.022, 0.925, 'Work in progress')

  if save:
    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
        print('Creating folder plots')
    plt.savefig('plots/'+fileName+'_TrainTestScore' + addStr + '.pdf')
    plt.savefig('plots/'+fileName+'_TrainTestScore' + addStr + '.png')
    plt.close()

