#!/usr/bin/env python

import os, sys

import matplotlib
import matplotlib.pyplot as plt

import AtlasStyle_mpl
import ROOT

SAVEDIR = '/project/etp5/dhandl/plots/Stop1L/FullRun2/FitResults/shapeFit_allSyst_JESreduced2_18032019/pdf/'
if not os.path.exists(SAVEDIR):
  os.makedirs(SAVEDIR)

FILENAME = 'cls_comparison'

LUMI = 140500.

logScale = False
if logScale:
  FILENAME = FILENAME + '_log'


def normQuantileHack(p):
  _p = p
  if p < 3e-15: # NormQuantile will spit out 0 then ...
      #logger.warn("Too small p-value for ROOT.TMath.NormQuantile - will set it to 3e-15 (~7.8 sigma)")
      print "Too small p-value for ROOT.TMath.NormQuantile - will set it to 3e-15 (~7.8 sigma)"
      _p = 3e-15
  return ROOT.TMath.NormQuantile(1-_p)

signal = [
      {'name':'bWN_650_500', 'legend':r'$m(\tilde{t}_1,\tilde{\chi}_{1}^{0})=(650,500)$', 'cls':[0.16292,0.0930293,0.0834068,0.0833281,0.0829448,0.0820409,0.0818596,0.0677506,0.067694,0.0685673], 'color':'red'},
      {'name':'bWN_650_560', 'legend':r'$m(\tilde{t}_1,\tilde{\chi}_{1}^{0})=(650,560)$', 'cls':[0.338679,0.338451,0.337694,0.335588,0.331297,0.328854,0.329449,0.29463,0.291933,0.295238], 'color':'blue'}
]

def plot():
  print 'test' 
  for s in signal:
    s['p'] = []
    for i in s['cls']:
      s['p'].append(normQuantileHack(i))

  print('Plotting p-value ...')
  fig = plt.figure(figsize=(8,6))
  ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=4)
  ax1.set_xlabel('Number of bins', horizontalalignment='right', x=1.0)
  ax1.set_ylabel('p', horizontalalignment='right', y=1.0)

  for s in signal:
    if logScale:
      ax1.set_yscale('log')
    plt.plot([1,2,3,4,5,6,7,8,9,10], s['p'], 'o-', color=s['color'], label=s['legend'], lw=2)
    ax1.set_xlim((1, 10))
    ax1.set_ylim((0, 2.5))

  leg = plt.legend(loc="upper right", frameon=False)

  #AtlasStyle_mpl.ATLASLabel(ax1, 0.02, 0.25, 'Work in progress')
  AtlasStyle_mpl.Text(ax1, 0.15, 0.83, 'Simulation')
  AtlasStyle_mpl.LumiLabel(ax1, 0.15, 0.77, lumi=LUMI*0.001)

  plt.savefig(SAVEDIR+FILENAME+'.pdf')
  plt.savefig(SAVEDIR+FILENAME+'.png')
  plt.close()

if __name__ == "__main__":
  plot()
