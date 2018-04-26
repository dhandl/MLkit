#!/usr/bin/env python

import os, sys

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate

import AtlasStyle_mpl

from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import ROOT
ROOT.gSystem.Load('libRooStats')

from prepareTraining import loadDataFrame, weightFrame, selectVarList, applyCut, varHasIndex, pickIndex

from collections import namedtuple
Sample = namedtuple('Sample', 'name' 'path')

inputDir = '/project/etp5/dhandl/samples/SUSY/Stop1L/hdf5/cut_mt30_met60_preselection/'
modelDir = 'TrainedModels/models/2018-04-12_10-21_DNN_ADAM_layer3x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'

SIGNAL = ['stop_bWN_250_100', 'stop_bWN_300_150', 'stop_bWN_350_200', 'stop_bWN_400_250', 'stop_bWN_450_300', 'stop_bWN_500_350', 'stop_bWN_550_400', 'stop_bWN_600_450', 'stop_bWN_650_500']

BACKGROUND = ['powheg_ttbar', 'powheg_singletop', 'sherpa22_Wjets']

PRESELECTION = [
                {'name':'n_jet',  'threshold':4,      'type':'geq'},
                {'name':'n_bjet',  'threshold':1,      'type':'geq'},
                {'name':'met',    'threshold':100e3,  'type':'geq'},
                {'name':'mt',    'threshold':90e3,  'type':'geq'},
                {'name':'n_lep',  'threshold':1,      'type':'exact'}
               ]

VAR = [
        'n_jet',
        'ht',
        'jet_pt[0]',
        'bjet_pt[0]',
        'jet_pt[1]',
        'jet_pt[2]',
        'jet_pt[3]',
        'lep_pt[0]',
        'amt2',
        'mt',
        'met',
        'dphi_met_lep',
        'met_sig',
        'ht_sig',
        'm_bl',
        'dr_bjet_lep',
        'mT_blMET'
      ]

WEIGHTS = [
           'weight',
           'xs_weight',
           'sf_total',
           'weight_sherpa22_njets'
          ]

LUMI = 140e3

RESOLUTION = np.array([50,0,1], dtype=float)

SCALING = './test.pkl'

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
  
  model = load_model(modelDir)

  scaler = joblib.load(SCALING)

  db = (RESOLUTION[2] - RESOLUTION[1]) / RESOLUTION[0]    # bin width in discriminator distribution
  bins = np.arange(RESOLUTION[1], RESOLUTION[2]+db, db)   # bin edges in discriminator distribution

  ###########################
  # Read and evaluate signals
  ###########################

  Signal = []
  for s in SIGNAL:
    x, y = pickBenchmark(s)
    df, weight = loadDataFrame(os.path.join(inputDir, s+'/'), PRESELECTION, VAR, WEIGHTS, LUMI)
    y_hat = evaluate(model, df.values, scaler)
    bin_index = np.digitize(y_hat[:,0], bins[1:])   # get the bin index of the output score for each event 
    outputWeighted = []
    outputWeightedVar = []
    outputMC = []
    outputMCVar = []
    for i in range(len(bins[1:])):
      w = weight.values[np.where(bin_index==i)[0]]
      sigma = np.sum(w**2.)
      outputWeighted.append(w.sum())
      outputWeightedVar.append(sigma)
      outputMC.append(len(w))
      outputMCVar.append(np.sqrt(len(w)))
    
    Signal.append({'name':s, 'm_stop':x, 'm_X':y, 'dataset':df, 'weight':weight, 'nEvents':weight.sum(), 'y_pred':y_hat, 'outputScore':np.array(outputWeighted), 'outputMC':np.array(outputMC), 'output_var':np.array(outputWeightedVar), 'outputMC_var':np.array(outputMCVar)})

    del df, weight, y_hat, bin_index, outputWeighted, outputWeightedVar, outputMC, outputMCVar

  ###########################
  # Read and evaluate backgrounds 
  ###########################
  
  totBkgEvents = 0.
  totBkgVar = 0.
  Background = []
  for b in BACKGROUND:
    df, weight = loadDataFrame(os.path.join(inputDir, b+'/'), PRESELECTION, VAR, WEIGHTS, LUMI)
    y_hat = evaluate(model, df.values, scaler)
    bin_index = np.digitize(y_hat[:,0], bins[1:])
    outputWeighted = []
    outputWeightedVar = []
    outputMC = []
    outputMCVar = []

    totBkgEvents += weight.sum()
    totBkgVar += np.sum(weight.values**2.)
    for i in range(len(bins[1:])):
      w = weight.values[np.where(bin_index==i)[0]]
      sigma = np.sum(w**2.)
      outputWeighted.append(w.sum())
      outputWeightedVar.append(sigma)
      outputMC.append(len(w))
      outputMCVar.append(len(w))

    Background.append({'name':b, 'dataset':df, 'weight':weight, 'nEvents':weight.sum(), 'y_pred':y_hat, 'outputScore':np.array(outputWeighted), 'outputMC':np.array(outputMC), 'output_var':np.array(outputWeightedVar), 'outputMC_var':np.array(outputMCVar)})

    del df, weight, y_hat, bin_index, outputWeighted, outputWeightedVar, outputMC, outputMCVar
  
  totalBkgOutput = np.array([b['outputScore'] for b in Background]) 
  totalBkgOutput = totalBkgOutput.sum(axis=0)
  
  totalBkgVar = np.array([b['output_var'] for b in Background])
  totalBkgVar = totalBkgVar.sum(axis=0)
   
  for s in Signal:
    significance = []
    significance_err = []
    tot_rel = np.sqrt(np.sum(s['output_var'])) / s['nEvents']
    for i in range(len(bins[1:])):
      #eff_sig = s['outputScore'][:i+1].sum() / s['nEvents']
      #eff_bkg = totalBkgOutput[:i+1].sum() / totalBkgOutput.sum()
      eff_sig = s['outputScore'][i:-1].sum() / s['nEvents']
      eff_bkg = totalBkgOutput[i:-1].sum() / totalBkgOutput.sum()
 
      #err_sig = np.sqrt(np.sum(s['output_var'][:i+1])) / s['nEvents']
      #err_bkg = np.sqrt(np.sum(totalBkgVar[:i+1])) / totalBkgOutput.sum()
      err_sig = np.sqrt(np.sum(s['output_var'][i:-1])) / s['nEvents']
      err_bkg = np.sqrt(np.sum(totalBkgVar[i:-1])) / totalBkgOutput.sum()

      #if totalBkgOutput[:i+1].sum() > 0.:
      #  rel_err_bkg = np.sqrt(np.sum(totalBkgVar[:i+1])) / totalBkgOutput[:i+1].sum()
      if totalBkgOutput[i:-1].sum() > 0.:
        rel_err_bkg = np.sqrt(np.sum(totalBkgVar[i:-1])) / totalBkgOutput[i:-1].sum()
      else:
        rel_err_bkg = 0.
      #if s['outputScore'][:i+1].sum() > 0.:
      #  rel_err_sig = np.sqrt(np.sum(s['output_var'][:i+1])) / s['outputScore'][:i+1].sum()
      if s['outputScore'][i:-1].sum() > 0.:
        rel_err_sig = np.sqrt(np.sum(s['output_var'][i:-1])) / s['outputScore'][i:-1].sum()
      else:
        rel_err_sig = 0.
      
      total_rel_err = np.sqrt(rel_err_sig**2. + rel_err_bkg**2. + 0.25**2.)

      if (eff_sig == 0) or (eff_bkg == 0):
        Z = 0.
        Z_err = 0.
      elif (err_sig / eff_sig > 0.75) or (err_bkg / eff_bkg > 0.75):
        Z = 0
        Z_err = 0
      else:
        #Z = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(s['outputScore'][:i+1].sum(), totalBkgOutput[:i+1].sum(), total_rel_err)
        Z = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(s['outputScore'][i:-1].sum(), totalBkgOutput[i:-1].sum(), total_rel_err)

        Zplus_sig = ROOT.RooStats.NumberCountingUtils.BinomialExpZ((eff_sig + err_sig) * s['nEvents'], eff_bkg * totalBkgOutput.sum(), total_rel_err)
        Zmins_sig = ROOT.RooStats.NumberCountingUtils.BinomialExpZ((eff_sig - err_sig) * s['nEvents'], eff_bkg * totalBkgOutput.sum(), total_rel_err)
        Zplus_bkg = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(eff_sig * s['nEvents'], (eff_bkg + err_bkg) * totalBkgOutput.sum(), total_rel_err)
        Zmins_bkg = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(eff_sig * s['nEvents'], (eff_bkg - err_bkg) * totalBkgOutput.sum(), total_rel_err)

      Z_err_sig = abs(Zplus_sig - Zmins_sig) / 2
      Z_err_bkg = abs(Zplus_bkg - Zmins_bkg) / 2
      Z_err = np.sqrt(Z_err_sig**2 + Z_err_bkg**2)

      significance.append(Z)
      significance_err.append(Z_err)

    s['sig'] = np.array(significance)
    s['sig_max'] = s['sig'].max()
    s['sig_err'] = np.array(significance_err)
    print s['sig']
    print s['sig'].max(), bins[np.where(s['sig'] == s['sig'].max())]

  x = np.array([s['m_stop'] for s in Signal], dtype=float)
  y = np.array([s['m_X'] for s in Signal], dtype=float)
  z = np.array([s['sig_max'] for s in Signal],dtype=float)

  print x, y, z
  # Set up a regular grid of interpolation points
  fig = plt.figure(figsize=(8,6))
  ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=4)
  xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
  xi, yi = np.meshgrid(xi, yi)

  # Interpolate
  rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
  zi = rbf(xi, yi)

  plt.imshow(zi, vmin=0., vmax=5., origin='lower',
             extent=[x.min(), x.max(), y.min(), y.max()])
  cbar = plt.colorbar()
  cbar.set_label('Significance')
  plt.scatter(x, y, c='black')
  plt.plot(x, x-84., color='black')
  plt.plot(x, x-175., color='black')
  ax1.set_ylabel(r'$m_{\chi}$')
  ax1.set_ylabel(r'$m_{\tilde{t}}$')
  AtlasStyle_mpl.ATLASLabel(ax1, 0.02, 0.9, 'Work in progress')
  AtlasStyle_mpl.LumiLabel(ax1, 0.02, 0.8, lumi=LUMI*0.001)
  plt.show()

if __name__ == "__main__":
    main()

