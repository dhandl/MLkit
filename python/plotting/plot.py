import os

import pandas as pd
import numpy as np

from getRatio import getRatio
from getSumW2 import getSumW2

#matplot lib for plotting
import matplotlib
import matplotlib.pyplot as plt

import AtlasLabel_mpl

def plotShape(var, samples, weights, color, binning, xTitle, yTitle="Events", lumi=100, unit=None, legend=None, log=False, ratio=False, ratioTitle='1/nominal', ratioLimit=(0,2) ,normed=False, savePlot=False, fileName=None):

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

  if (unit == None) or (unit.lower() == 'mev'):
    unit_fact = 1.
  elif (unit.lower() == 'gev'):
    unit_fact = 0.001

  if not type(samples) == list:
    if not type(samples) == tuple:
      print "Expected {} sample as tuple of variables and weights!".format(samples)
      return 0

    sumW2 = getSumW2(samples[0][str(var)].ravel(), samples[1].ravel(), binning)

    hist, bins, patches = np.histgram(samples[0][str(var)].ravel()*unit_fact, weights=samples[1].ravel(), bins=binning[0], range=(binning[1], binning[2]), density=normed)

    width = bins[1] - bins[0]
    center = (bins[:-1] + bins[1:]) / 2

    plt.errorbar(center, hist, xerr=[width/2.]*binning[0], yerr=sumW2.ravel(), fmt='o', color=color, label=legend)

    _max = hist.max()

  else:
    sumW2 = []
    hists = []

    for i, smp in enumerate(samples):
      #if not type(smp) == tuple:
      #  print "Expected {} sample as tuple of variables and weights!".format(smp)
      #  return 0

      sumW2.append(getSumW2(smp[str(var)].ravel(), weights[i].ravel(), binning))

      hists.append(np.histogram(smp[str(var)].ravel()*unit_fact, weights=weights[i], bins=binning[0], range=(binning[1], binning[2]), density=normed))

      width = hists[i][1][1] - hists[i][1][0]
      center = (hists[i][1][:-1] + hists[i][1][1:]) / 2

      plt.errorbar(center, hists[i][0], xerr=[width/2.]*binning[0], yerr=sumW2[i].ravel(), fmt='o', color=color[i], label=legend[i])

    _max = np.max([h[0].max() for h in hists])

  if normed:
    ax1.set_ylabel("a. u.", va='top')
  else:
    ax1.set_ylabel("Events", va='top') 

  if log:
    ax1.set_yscale('log')
    ax1.set_ylim((0.01, _max*100))
  else:
    if normed:
      ax1.set_ylim((0, 1.5))
    else:
      ax1.set_ylim((0, _max*1.4))

  leg = plt.legend(loc='best', frameon=False)

  AtlasLabel_mpl.ATLASLabel(ax1, 0.02, 0.9, 'Work in progress')
  AtlasLabel_mpl.LumiLabel(ax1, 0.02, 0.8, lumi=str(lumi))

  if ratio:
    ax2 = plt.subplot2grid((4,4), (3,0), colspan=4, rowspan=1)

    for i in range(1, len(hists)):
      r = getRatio(hists[i][0], hists[i][1], sumW2[i], hists[0][0], hists[0][1], sumW2[0], color[i])

    ax2.set_xlabel(xTitle, ha='right')
    ax2.set_ylabel(ratioTitle, va='top')
    ax2.set_xlim((binning[1],binning[2]))
    ax2.set_ylim(ratioLimit)
    ax2.grid()
    ax2.tick_params(direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')

  ax1.set(xlabel=xTitle)

  if savePlot:
    plt.savefig(fileName+".pdf")
    plt.savefig(fileName+".png")
    plt.close()

#  elif type(samples) == tuple:
#    if (unit == None) or (unit.lower() == 'gev'):
#      hist, bins, patches = plt.hist(samples[0][var], bins=binning[0], range=(binning[1],binning[2]),\
#                                     normed=normed, weights=samples[1], cumulative=False, bottom=None, histtype='step',\
#                                     align='mid', orientation='vertical', rwidth=None, log=log, color=colors[0],\
#                                     label=legend, stacked=False, hold=None, data=None)
#    elif unit.lower() == "mev":
#      hist, bins, patches = plt.hist(samples[0][var]*0.001, bins=binning[0], range=(binning[1],binning[2]),\
#                                     normed=normed, weights=samples[1], cumulative=False, bottom=None, histtype='step',\
#                                     align='mid', orientation='vertical', rwidth=None, log=log, color=colors[0],\
#                                     label=legend, stacked=False, hold=None, data=None)
#  elif type(samples) == list:
#    if (unit == None) or (unit.lower() == 'gev'):
#      hist, bins, patches = plt.hist([s[0][var] for s in samples], bins=binning[0], range=(binning[1],binning[2]),\
#                                     normed=normed, weights=[s[1] for s in samples], cumulative=False, bottom=None, histtype='step',\
#                                     align='mid', orientation='vertical', rwidth=None, log=log, color=[colors[i] for i in range(len(samples))],\
#                                     label=legend, stacked=False, hold=None, data=None)
#    elif unit.lower() == "mev":
#      hist, bins, patches = plt.hist([s[0][var]*0.001 for s in samples], bins=binning[0], range=(binning[1],binning[2]),\
#                                     normed=normed, weights=[s[1] for s in samples], cumulative=False, bottom=None, histtype='step',\
#                                     align='mid', orientation='vertical', rwidth=None, log=log, color=[colors[i] for i in range(len(samples))],\
#                                     label=legend, stacked=False, hold=None, data=None)
#  plt.subplots_adjust(left=0.15)
#  plt.xlabel(xTitle)
#  if normed:
#    plt.ylabel("a. u.")
#  else:
#    plt.ylabel(yTitle)
#  plt.legend(legend, loc='best')
#  if save:
#    plt.savefig(fileName+".pdf")
#    plt.savefig(fileName+".png")
#    plt.close()
#
