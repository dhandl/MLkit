import os

from separation import getSeparation
from getRatio import getRatio
from getSumW2 import getSumW2
# arrays
import numpy as np
import pandas as pd

# matplot lib for plotting
import matplotlib
import matplotlib.pyplot as plt

import AtlasStyle_mpl

def plot_output_score(sig_predicted, sig_w, bkg_predicted, bkg_w, binning, fileName="Test", normed=False, save=False, ratio=True):
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

  s_hist, s_bins, s_patches = plt.hist(sig_predicted.ravel(), weights=sig_w, histtype='stepfilled', color='r', label='ttbar nominal', alpha=0.5, bins=binning[0], range=(binning[1], binning[2]), density=normed)
  b_hist, b_bins, b_patches = plt.hist(bkg_predicted.ravel(), weights=bkg_w, histtype='stepfilled', color='b', label='ttbar radiation low', alpha=0.5, bins=binning[0], range=(binning[1], binning[2]), density=normed)

  s_w = getSumW2(sig_predicted.ravel(), sig_w, binning)
  b_w = getSumW2(bkg_predicted.ravel(), bkg_w, binning)

  #sep = getSeparation(s_histTest, s_binsTest, b_histTest, b_binsTest)

  #print sep

  if normed:
    ax1.set_ylabel("a. u.", ha='left')
  else:
    ax1.set_ylabel("Events", ha='left')
  
  ax1.set_ylim((0, s_hist.max()*(1+0.33)))
  leg = plt.legend(loc="best", frameon=False)
  p = leg.get_window_extent()
  #ax.annotate('KS Test S (B): %.3f (%.3f)'%(ks_sig, ks_bkg),(p.p0[0], p.p1[1]), (p.p0[0], p.p1[1]), xycoords='figure pixels', zorder=9)
  #ax1.text(0.65, 0.7, "KS Test S (B): %.3f (%.3f)"%(ks_sig, ks_bkg), transform=ax1.transAxes)
  #ax1.text(0.65, 0.70, '$<S^2>$ = %.3f'%(sep), transform=ax1.transAxes)
  #ax.text(0.55, 0.7, "KS p-value S (B): %.3f (%.3f)"%(ks_sig_p, ks_bkg_p), transform=ax.transAxes)

  AtlasStyle_mpl.ATLASLabel(ax1, 0.02, 0.9, 'Work in progress')
  AtlasStyle_mpl.LumiLabel(ax1, 0.02, 0.8, lumi=100)
  if ratio:
    ax2 = plt.subplot2grid((4,4), (3,0), colspan=4, rowspan=1)
    r = getRatio(b_hist, b_bins, b_w, s_hist, s_bins, s_w, 'r')
    ax2.set_xlabel('Discriminant')
    ax2.set_ylabel('variation/nom.')
    ax2.set_xlim((binning[1],binning[2]))
    ax2.set_ylim((-0.5,2.5))
    ax2.grid()
    ax2.tick_params(direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')

  ax1.set(xlabel='Output score')

  if save:
    if not os.path.exists("./plots/"):
        os.makedirs("./plots/")
        print("Creating folder plots")
    plt.savefig("plots/"+fileName+"_output_score.pdf")
    plt.savefig("plots/"+fileName+"_output_score.png")
    plt.close()
  return r, s_bins
