#!/usr/bin/env python

import os, sys

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.interpolate

import AtlasStyle_mpl

from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc, roc_auc_score

import ROOT
ROOT.gSystem.Load('libRooStats')

from prepareTraining import loadDataFrame, weightFrame, selectVarList, applyCut, varHasIndex, pickIndex
from getRatio import getRatio

from collections import namedtuple
Sample = namedtuple('Sample', 'name' 'path')

sys.path.append('./python/plotting/')
import plot_ROCcurves

#inputDir = '/project/etp5/dhandl/samples/SUSY/Stop1L/hdf5/cut_mt30_met60_preselection/'
inputDir = '/project/etp5/dhandl/samples/SUSY/Stop1L/FullRun2/hdf5/cut_mt30_met60_preselection/'

DIR = 'TrainedModels/models/'

FILENAME = 'TAT_similarStats'

logScale = True
if logScale:
  FILENAME = FILENAME + '_log'

models = [
  {'name':'TRUTH', 'mdir':DIR+'2018-08-02_15-24_DNN_ADAM_layer1x100_batch10_NormalInitializer_dropout0p5_l1-0p01.h5', 'sdir':DIR+'2018-08-02_15-24_DNN_ADAM_layer1x100_batch10_NormalInitializer_dropout0p5_l1-0p01_scaler.pkl'},
  {'name':'RECO',  'mdir':DIR+'2018-08-02_16-03_DNN_ADAM_layer1x100_batch10_NormalInitializer_dropout0p5_l1-0p01.h5', 'sdir':DIR+'2018-08-02_16-03_DNN_ADAM_layer1x100_batch10_NormalInitializer_dropout0p5_l1-0p01_scaler.pkl'}
]

#SIGNAL = ['stop_bWN_250_100', 'stop_bWN_250_130', 'stop_bWN_300_150', 'stop_bWN_300_180', 'stop_bWN_350_200', 'stop_bWN_350_230', 'stop_bWN_400_250', 'stop_bWN_400_280', 'stop_bWN_450_300', 'stop_bWN_450_330', 'stop_bWN_500_350', 'stop_bWN_500_380', 'stop_bWN_550_400', 'stop_bWN_550_430', 'stop_bWN_600_450', 'stop_bWN_600_480', 'stop_bWN_650_500', 'stop_bWN_650_530']
SIGNAL = ['stop_bWN_450_300_mc16d']

BACKGROUND = ['mc16d_ttbar', 'mc16d_singletop', 'mc16d_Wjets']

PRESELECTION = [
                {'name':'n_jet',  'threshold':4,      'type':'geq'},
                {'name':'n_bjet',  'threshold':1,      'type':'geq'},
                {'name':'met',    'threshold':100e3,  'type':'geq'},
                {'name':'mt',    'threshold':110e3,  'type':'geq'},
                {'name':'n_lep',  'threshold':1,      'type':'exact'}
               ]

VAR = [
        #'bjet_pt[0]', 'amt2', 'mt', 'met', 'dphi_met_lep', 'ht_sig', 'dr_bjet_lep'
        #'met',
        #'mt',
        #'Lp',
        #'dphi_met_lep',
        #'met_sig',
        #'m3'
        'met',
        'met_phi',
        'n_jet',
        'n_bjet',
        'jet_pt[0]',
        'jet_eta[0]',
        'jet_phi[0]',
        'jet_e[0]',
        #'jet_bweight[0]',
        'jet_pt[1]',
        'jet_eta[1]',
        'jet_phi[1]',
        'jet_e[1]',
        #'jet_bweight[1]',
        'jet_pt[2]',
        'jet_eta[2]',
        'jet_phi[2]',
        'jet_e[2]',
        #'jet_bweight[2]',
        'jet_pt[3]',
        'jet_eta[3]',
        'jet_phi[3]',
        'jet_e[3]',
        #'jet_bweight[3]'
        'lep_pt[0]',
        'lep_eta[0]',
        'lep_phi[0]',
        'lep_e[0]'
     ]

WEIGHTS = [
           'weight',
           'xs_weight',
           'sf_total',
           'weight_sherpa22_njets'
          ]

LUMI = 140e3

RESOLUTION = np.array([50,0,1], dtype=float)
db = (RESOLUTION[2] - RESOLUTION[1]) / RESOLUTION[0]    # bin width in discriminator distribution
bins = np.arange(RESOLUTION[1], RESOLUTION[2]+db, db)   # bin edges in discriminator distribution
center = (bins[:-1] + bins[1:]) / 2

def asimovZ(s, b, b_err, syst=False):
  tot = s + b
  b2 = b*b
  if syst:
    b_err2 = np.sqrt( b_err*b_err + (b*0.25)*(b*0.25) )
  else:
    b_err2 = b_err * b_err
  b_plus_err2 = b + b_err2
  Z = np.sqrt(2 * ((tot)*np.log(tot * b_plus_err2 / (b2 + tot * b_err2)) - b2 / b_err2 * np.log(1 + b_err2 * s / (b * b_plus_err2))))
  return Z

def evaluate(model, dataset, scaler):
  dataset = scaler.transform(dataset)
  y_hat = model.predict(dataset)
  return y_hat

def pickBenchmark(signal, delimiter='_'):
  try:
    name = signal.split(delimiter)
    x = name[2]
    y = name[3]
    return x, y
  except Exception:
    print 'ERROR: No matching pattern for benchmark {}'.format(signal)
    return 0

def main():
 
  for m in models:
    m['model'] = load_model(m['mdir'])
    m['scaler'] = joblib.load(m['sdir'])


  ###########################
  # Read and evaluate signals
  ###########################

  Signal = []
  for s in SIGNAL:
    x, y = pickBenchmark(s)
    df, weight = loadDataFrame(os.path.join(inputDir, s+'/'), PRESELECTION, VAR, WEIGHTS, LUMI)
    for m in models:
      m['y_pred_sig'] = evaluate(m['model'], df.values, m['scaler'])
      m['y_sig'] = np.ones(m['y_pred_sig'].shape[0])
    #bin_index = np.digitize(y_hat[:,0], bins[1:])   # get the bin index of the output score for each event 
    #outputWeighted = []
    #outputWeightedVar = []
    #outputMC = []
    #outputMCVar = []
    #for i in range(len(bins[1:])):
    #  w = weight.values[np.where(bin_index==i)[0]]
    #  sigma = np.sum(w**2.)
    #  outputWeighted.append(w.sum())
    #  outputWeightedVar.append(sigma)
    #  outputMC.append(len(w))
    #  outputMCVar.append(np.sqrt(len(w)))
    #
    #Signal.append({'name':s, 'm_stop':x, 'm_X':y, 'dataset':df, 'weight':weight, 'nEvents':weight.sum(), 'outputScore':np.array(outputWeighted), 'outputMC':np.array(outputMC), 'output_var':np.array(outputWeightedVar), 'outputMC_var':np.array(outputMCVar)})

    #del df, weight, y_hat, bin_index, outputWeighted, outputWeightedVar, outputMC, outputMCVar

  ###########################
  # Read and evaluate backgrounds 
  ###########################
  
  #totBkgEvents = 0.
  #totBkgVar = 0.
  #Background = []
  for b in BACKGROUND:
    df, weight = loadDataFrame(os.path.join(inputDir, b+'/'), PRESELECTION, VAR, WEIGHTS, LUMI)
    for m in models:
      m['_'.join(['y_pred',b])] = evaluate(m['model'], df.values, m['scaler'])
      m['_'.join(['y',b])] = np.zeros(m['_'.join(['y_pred',b])].shape[0])
    #bin_index = np.digitize(y_hat[:,0], bins[1:])
    #outputWeighted = []
    #outputWeightedVar = []
    #outputMC = []
    #outputMCVar = []

    #totBkgEvents += weight.sum()
    #totBkgVar += np.sum(weight.values**2.)
    #for i in range(len(bins[1:])):
    #  w = weight.values[np.where(bin_index==i)[0]]
    #  sigma = np.sum(w**2.)
    #  outputWeighted.append(w.sum())
    #  outputWeightedVar.append(sigma)
    #  outputMC.append(len(w))
    #  outputMCVar.append(len(w))

    #Background.append({'name':b, 'dataset':df, 'weight':weight, 'nEvents':weight.sum(), 'y_pred':y_hat, 'outputScore':np.array(outputWeighted), 'outputMC':np.array(outputMC), 'output_var':np.array(outputWeightedVar), 'outputMC_var':np.array(outputMCVar)})

    #del df, weight, y_hat, bin_index, outputWeighted, outputWeightedVar, outputMC, outputMCVar
  
  for m in models:
    m['y_bkg'] = np.empty(0)
    m['y_pred_bkg'] = np.empty(0)

    for b in BACKGROUND:
      m['y_bkg'] = np.concatenate((m['y_bkg'], m['_'.join(['y',b])])) 
      m['y_pred_bkg'] = np.concatenate((m['y_pred_bkg'], m['_'.join(['y_pred',b])][:,0]))

    m['y'] = np.concatenate((m['y_sig'], m['y_bkg']))
    m['y_pred'] = np.concatenate((m['y_pred_sig'][:,0], m['y_pred_bkg']))

    m['fpr'], m['tpr'], m['threshold'] = roc_curve(m['y'], m['y_pred'])
    m['auc'] = roc_auc_score(m['y'], m['y_pred']) 

  print('Plotting ROC curve ...')
  fig = plt.figure(figsize=(8,6))
  ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=4)
  ax1.set_xlabel('Signal efficiency', horizontalalignment='right', x=1.0)
  ax1.set_ylabel('Background rejection', horizontalalignment='right', y=1.0)

  for m in models:
    if logScale:
      ax1.set_yscale('log')
      plt.plot(m['tpr'], 1./m['fpr'], lw=2, label=m['name']+' (AUC = %0.3f)'%(m['auc'])) 
    else:
      plt.plot(m['tpr'], 1.-m['fpr'], lw=2, label=m['name']+' (AUC = %0.3f)'%(m['auc']))
      ax1.set_xlim((0, 1))
      ax1.set_ylim((0, 1))
      #plt.plot([0, 1], [1, 0], '--', color=(0.6, 0.6, 0.6), label='Luck')

  leg = plt.legend(loc="lower left", frameon=False)

  AtlasStyle_mpl.ATLASLabel(ax1, 0.02, 0.25, 'Work in progress')

  plt.savefig("plots/"+FILENAME+'.pdf')
  plt.savefig("plots/"+FILENAME+'.png')
  plt.close()

if __name__ == "__main__":
    main()
