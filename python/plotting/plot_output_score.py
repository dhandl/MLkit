import os

from separation import getSeparation
from getRatio import getRatio
# arrays
import numpy as np
import pandas as pd

# matplot lib for plotting
import matplotlib
import matplotlib.pyplot as plt

def plot_output_score(sig_predicted, sig_w, bkg_predicted, bkg_w, binning, fileName=None, normed=False, save=False, ratio=True):
  fig = plt.figure(figsize=(8,6))
  if ratio:
    ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=3)
    ax1.set_xlabel('', fontsize=0.)
    ax1.set_xticklabels(())
  else: 
    ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=4)
  ax1.tick_params(direction='in')
  ax1.set_xlim((binning[1], binning[2]))
  ax1.xaxis.set_ticks_position('both')
  ax1.yaxis.set_ticks_position('both')

  s_hist, s_bins, s_patches = plt.hist(sig_predicted.ravel(), weights=sig_w, histtype='stepfilled', color='r', label='ttbar nominal', alpha=0.5, bins=binning[0], range=(binning[1], binning[2]), normed=normed)
  b_hist, b_bins, b_patches = plt.hist(bkg_predicted.ravel(), weights=bkg_w, histtype='stepfilled', color='b', label='ttbar radiation low', alpha=0.5, bins=binning[0], range=(binning[1], binning[2]), normed=normed)

  #sep = getSeparation(s_histTest, s_binsTest, b_histTest, b_binsTest)

  #print sep

  if normed:
    ax1.set(ylabel="a. u.")
  else:
    ax1.set(ylabel="Events")
  
  ax1.set_ylim((0, s_hist.max()*(1+0.2)))
  leg = plt.legend(loc="best", frameon=False)
  p = leg.get_window_extent()
  #ax.annotate('KS Test S (B): %.3f (%.3f)'%(ks_sig, ks_bkg),(p.p0[0], p.p1[1]), (p.p0[0], p.p1[1]), xycoords='figure pixels', zorder=9)
  #ax1.text(0.65, 0.7, "KS Test S (B): %.3f (%.3f)"%(ks_sig, ks_bkg), transform=ax1.transAxes)
  #ax1.text(0.65, 0.70, '$<S^2>$ = %.3f'%(sep), transform=ax1.transAxes)
  #ax.text(0.55, 0.7, "KS p-value S (B): %.3f (%.3f)"%(ks_sig_p, ks_bkg_p), transform=ax.transAxes)

  if ratio:
    ax2 = plt.subplot2grid((4,4), (3,0), colspan=4, rowspan=1)
    getRatio(s_hist, s_bins, b_hist, b_bins)
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

