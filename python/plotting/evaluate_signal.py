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
from getRatio import getRatio

from collections import namedtuple
Sample = namedtuple('Sample', 'name' 'path')

sys.path.append('./python/plotting/')
import plot_ROCcurves

#inputDir = '/project/etp5/dhandl/samples/SUSY/Stop1L/hdf5/cut_mt30_met60_preselection/'
inputDir = '/project/etp5/dhandl/samples/SUSY/Stop1L/FullRun2/hdf5/cut_mt30_met60_preselection/'

Dir = 'TrainedModels/models/'
#modelfile = '2018-07-31_11-18_DNN_ADAM_layer1x100_batch40_NormalInitializer_dropout0p5_l1-0p01_multiclass'
#modelfile = '2018-07-31_10-52_DNN_ADAM_layer1x100_batch40_NormalInitializer_dropout0p5_l1-0p01_multiclass'
#modelfile = '2018-07-31_13-16_DNN_ADAM_layer1x100_batch40_NormalInitializer_dropout0p5_l1-0p01_multiclass'
modelfile = '2018-07-31_14-15_DNN_ADAM_layer1x100_batch40_NormalInitializer_dropout0p5_l1-0p01_multiclass'

modelDir = Dir+modelfile+'.h5'

#SIGNAL = ['stop_bWN_250_100', 'stop_bWN_250_130', 'stop_bWN_300_150', 'stop_bWN_300_180', 'stop_bWN_350_200', 'stop_bWN_350_230', 'stop_bWN_400_250', 'stop_bWN_400_280', 'stop_bWN_450_300', 'stop_bWN_450_330', 'stop_bWN_500_350', 'stop_bWN_500_380', 'stop_bWN_550_400', 'stop_bWN_550_430', 'stop_bWN_600_450', 'stop_bWN_600_480', 'stop_bWN_650_500', 'stop_bWN_650_530']
SIGNAL = ['stop_bWN_450_300_mc16d']
#SIGNAL = ['stop_tN_800_500']

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

SCALING = Dir+modelfile+'_scaler.pkl'

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
  
  model = load_model(modelDir)

  scaler = joblib.load(SCALING)

  db = (RESOLUTION[2] - RESOLUTION[1]) / RESOLUTION[0]    # bin width in discriminator distribution
  bins = np.arange(RESOLUTION[1], RESOLUTION[2]+db, db)   # bin edges in discriminator distribution
  center = (bins[:-1] + bins[1:]) / 2

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
    asimov = []
    tot_rel = np.sqrt(np.sum(s['output_var'])) / s['nEvents']
    for i in range(len(bins[1:])):
      #eff_sig = s['outputScore'][:i+1].sum() / s['nEvents']
      #eff_bkg = totalBkgOutput[:i+1].sum() / totalBkgOutput.sum()
      eff_sig = s['outputScore'][i:].sum() / s['nEvents']
      eff_bkg = totalBkgOutput[i:].sum() / totalBkgOutput.sum()
 
      #err_sig = np.sqrt(np.sum(s['output_var'][:i+1])) / s['nEvents']
      #err_bkg = np.sqrt(np.sum(totalBkgVar[:i+1])) / totalBkgOutput.sum()
      err_sig = np.sqrt(np.sum(s['output_var'][i:])) / s['nEvents']
      err_bkg = np.sqrt(np.sum(totalBkgVar[i:])) / totalBkgOutput.sum()

      #if totalBkgOutput[:i+1].sum() > 0.:
      #  rel_err_bkg = np.sqrt(np.sum(totalBkgVar[:i+1])) / totalBkgOutput[:i+1].sum()
      if totalBkgOutput[i:].sum() > 0.:
        rel_err_bkg = np.sqrt(np.sum(totalBkgVar[i:])) / totalBkgOutput[i:].sum()
      else:
        rel_err_bkg = 0.
      #if s['outputScore'][:i+1].sum() > 0.:
      #  rel_err_sig = np.sqrt(np.sum(s['output_var'][:i+1])) / s['outputScore'][:i+1].sum()
      if s['outputScore'][i:].sum() > 0.:
        rel_err_sig = np.sqrt(np.sum(s['output_var'][i:])) / s['outputScore'][i:].sum()
      else:
        rel_err_sig = 0.
      
      #total_rel_err = np.sqrt(rel_err_sig**2. + rel_err_bkg**2. + 0.25**2.)
      total_rel_err = np.sqrt(rel_err_bkg**2. + 0.25**2.)

      if (eff_sig == 0) or (eff_bkg == 0):
        Z = 0.
        Z_err = 0.
        ams = 0.
      elif (err_sig / eff_sig > 0.75) or (err_bkg / eff_bkg > 0.75):
        Z = 0.
        Z_err = 0.
        ams = 0.
      else:
        #Z = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(s['outputScore'][:i+1].sum(), totalBkgOutput[:i+1].sum(), total_rel_err)
        Z = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(s['outputScore'][i:].sum(), totalBkgOutput[i:].sum(), total_rel_err)
        ams = asimovZ( s['outputScore'][i:].sum(), totalBkgOutput[i:].sum(), np.sqrt(totalBkgVar[i:].sum()))

        Zplus_sig = ROOT.RooStats.NumberCountingUtils.BinomialExpZ((eff_sig + err_sig) * s['nEvents'], eff_bkg * totalBkgOutput.sum(), total_rel_err)
        Zmins_sig = ROOT.RooStats.NumberCountingUtils.BinomialExpZ((eff_sig - err_sig) * s['nEvents'], eff_bkg * totalBkgOutput.sum(), total_rel_err)
        Zplus_bkg = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(eff_sig * s['nEvents'], (eff_bkg + err_bkg) * totalBkgOutput.sum(), total_rel_err)
        Zmins_bkg = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(eff_sig * s['nEvents'], (eff_bkg - err_bkg) * totalBkgOutput.sum(), total_rel_err)

      Z_err_sig = abs(Zplus_sig - Zmins_sig) / 2
      Z_err_bkg = abs(Zplus_bkg - Zmins_bkg) / 2
      Z_err = np.sqrt(Z_err_sig**2 + Z_err_bkg**2)

      significance.append(Z)
      significance_err.append(Z_err)
      asimov.append(ams)

    s['sig'] = np.array(significance)
    s['sig_max'] = s['sig'].max()
    s['sig_err'] = np.array(significance_err)
    s['ams'] = np.array(asimov)
    print s['sig']
    print s['ams']
    sigMax_index = bins[np.where(s['sig'] == s['sig'].max())][0]
    Z = asimovZ(Signal[0]['outputScore'][np.where(bins[:-1] == sigMax_index)], totalBkgOutput[np.where(bins[:-1] == sigMax_index)], np.sqrt(totalBkgVar[np.where(bins[:-1] == sigMax_index)]), syst=False)
    Z_syst = asimovZ(Signal[0]['outputScore'][np.where(bins[:-1] == sigMax_index)], totalBkgOutput[np.where(bins[:-1] == sigMax_index)], np.sqrt(totalBkgVar[np.where(bins[:-1] == sigMax_index)]), syst=True)
    print s['sig'].max(), sigMax_index, Z, Z_syst

  x = np.array([s['m_stop'] for s in Signal], dtype=float)
  y = np.array([s['m_X'] for s in Signal], dtype=float)
  z = np.array([s['sig_max'] for s in Signal],dtype=float)

  #print x, y, z

  print Signal[0]['outputScore'][np.where(bins[:-1] >= sigMax_index)], Signal[0]['output_var'][np.where(bins[:-1] >= sigMax_index)]
  print totalBkgOutput[np.where(bins[:-1] >= sigMax_index)], totalBkgVar[np.where(bins[:-1] >= sigMax_index)]

  print Signal[0]['outputScore'], Signal[0]['output_var']
  print totalBkgOutput, totalBkgVar
  # Set up a regular grid of interpolation points

  print('Plotting the output score...')
  fig = plt.figure(figsize=(8,6))
  ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=4)
  ax1.set_xlim((bins[0], bins[-1]))
  ax1.set(xlabel='Output score')
  ax1.set_ylabel("Events", ha='left')

  sb_ratio = Signal[0]['outputScore'].sum()/totalBkgOutput.sum()
  #if sb_ratio < 0.2:
  #  #ATTENTION! Simplified error propagation (treated as uncorrelated)
  #  scaled = Signal[0]['outputScore'] / Signal[0]['outputScore'].sum() * totalBkgOutput.sum()
  #  scaled_var = scaled*scaled * ( (Signal[0]['output_var']/Signal[0]['outputScore'])**2 + (totalBkgVar.sum()/totalBkgOutput.sum())**2 + (Signal[0]['output_var'].sum()/Signal[0]['outputScore'].sum())**2 )
  #  scaled_label = 'Signal scaled to Bkg'
  #  
  #else:
  scaled = Signal[0]['outputScore']
  scaled_var = Signal[0]['output_var']
  scaled_label = 'Signal'

  plt.bar(center, totalBkgOutput, width=db, yerr=np.sqrt(totalBkgVar), color='b', alpha=0.5, error_kw=dict(ecolor='b', lw=1.5), label='Background')  
  plt.bar(center, Signal[0]['outputScore'], width=db, yerr= np.sqrt(Signal[0]['output_var']), label=Signal[0]['name'], color='r', alpha=0.5, error_kw=dict(ecolor='r', lw=1.5))  

  ax1.set_ylim((0.1, totalBkgOutput.max()*(15.)))
  ax1.set_yscale('log')
  leg = plt.legend(loc="best", frameon=False)

  AtlasStyle_mpl.ATLASLabel(ax1, 0.02, 0.925, 'Work in progress')
  AtlasStyle_mpl.LumiLabel(ax1, 0.02, 0.875, lumi=LUMI*0.001)

  plt.savefig("plots/"+modelfile+"_eval-bWN-450-300_outputScore.pdf")
  plt.savefig("plots/"+modelfile+"_eval-bWN-450-300_outputScore.png")
  plt.close()

  #plot_ROCcurves.plot_ROC(y_train, y_test, y_predict_train, y_predict_test, save=save, fileName=modelfile+'_eval-bWN-450-300')

if __name__ == "__main__":
    main()
    
#def evaluate_signalGrid(modelDir, resolution=np.array([50,0,1], dtype=float), save=False, fileName="Grid_test"):
#  print('Evaluating singal grid...')  
#    
#  infofile = open(modelDir.replace(".h5","_infofile.txt"))
#  infos = infofile.readlines()
#  
#  #Parse Strings for correct datatypes
#  
#  variables=infos[4].replace('Used variables for training: ','').replace('\n','').split()
#  weights=infos[5].replace('Used weights: ', '').replace('\n','').split()
#  preselection_raw=infos[6].replace('Used preselection: ', '').replace('; \n', '').split(';')
#  preselection=[]
#  for x in preselection_raw:
#      xdict = {}
#      xdict['name']= x.split()[0].split('-')[0]
#      xdict['threshold']= float(x.split()[1])
#      xdict['type'] = x.split()[3]
#      preselection.append(xdict)
#  lumi=float(infos[7].replace('Used Lumi: ','').replace('\n',''))
#  background=infos[9].replace('Used background files: ','').replace('; \n','').replace(' ','').split(';')
#  #signal=infos[8].replace('Used signal files: ','').replace('; \n','').replace(' ','').split(';')
#  
#  signal = ['stop_bWN_250_100', 'stop_bWN_250_130', 'stop_bWN_300_150', 'stop_bWN_300_180', 'stop_bWN_350_200', 'stop_bWN_350_230', 'stop_bWN_400_250', 'stop_bWN_400_280', 'stop_bWN_450_300', 'stop_bWN_450_330', 'stop_bWN_500_350', 'stop_bWN_500_380', 'stop_bWN_550_400', 'stop_bWN_550_430', 'stop_bWN_600_450', 'stop_bWN_600_480', 'stop_bWN_650_500', 'stop_bWN_650_530']
#   
#  #For Debugging
#  #print variables, type(variables)
#  #print weights, type(variables)
#  #print preselection, type(preselection[1])
#  #print lumi, type(lumi)
#  #print signal, type(signal)
#  #print background, type(background)
#   
#  #Get Scaler and model from modelDir
#   
#  model = load_model(modelDir)
#  
#  scalerDir=modelDir.replace('.h5','_scaler.pkl')
#  scaler=joblib.load(scalerDir)
#    
#  #Evaluate
#
#  db = (resolution[2] - resolution[1]) / resolution[0]    # bin width in discriminator distribution
#  bins = np.arange(resolution[1], resolution[2]+db, db)   # bin edges in discriminator distribution
#
#  ###########################
#  # Read and evaluate signals
#  ###########################
#
#  Signal = []
#  for s in signal:
#    x, y = pickBenchmark(s)
#    df, weight = loadDataFrame(os.path.join(inputDir, s+'/'), preselection, variables, weights, lumi)
#    y_hat = evaluate(model, df.values, scaler)
#    bin_index = np.digitize(y_hat[:,0], bins[1:])   # get the bin index of the output score for each event 
#    outputWeighted = []
#    outputWeightedVar = []
#    outputMC = []
#    outputMCVar = []
#    for i in range(len(bins[1:])):
#      w = weight.values[np.where(bin_index==i)[0]]
#      sigma = np.sum(w**2.)
#      outputWeighted.append(w.sum())
#      outputWeightedVar.append(sigma)
#      outputMC.append(len(w))
#      outputMCVar.append(np.sqrt(len(w)))
#    
#    Signal.append({'name':s, 'm_stop':x, 'm_X':y, 'dataset':df, 'weight':weight, 'nEvents':weight.sum(), 'y_pred':y_hat, 'outputScore':np.array(outputWeighted), 'outputMC':np.array(outputMC), 'output_var':np.array(outputWeightedVar), 'outputMC_var':np.array(outputMCVar)})
#
#    del df, weight, y_hat, bin_index, outputWeighted, outputWeightedVar, outputMC, outputMCVar
#
#  ###########################
#  # Read and evaluate backgrounds 
#  ###########################
#  
#  totBkgEvents = 0.
#  totBkgVar = 0.
#  Background = []
#  for b in background:
#    df, weight = loadDataFrame(os.path.join(inputDir, b+'/'), preselection, variables, weights, lumi)
#    y_hat = evaluate(model, df.values, scaler)
#    bin_index = np.digitize(y_hat[:,0], bins[1:])
#    outputWeighted = []
#    outputWeightedVar = []
#    outputMC = []
#    outputMCVar = []
#
#    totBkgEvents += weight.sum()
#    totBkgVar += np.sum(weight.values**2.)
#    for i in range(len(bins[1:])):
#      w = weight.values[np.where(bin_index==i)[0]]
#      sigma = np.sum(w**2.)
#      outputWeighted.append(w.sum())
#      outputWeightedVar.append(sigma)
#      outputMC.append(len(w))
#      outputMCVar.append(len(w))
#
#    Background.append({'name':b, 'dataset':df, 'weight':weight, 'nEvents':weight.sum(), 'y_pred':y_hat, 'outputScore':np.array(outputWeighted), 'outputMC':np.array(outputMC), 'output_var':np.array(outputWeightedVar), 'outputMC_var':np.array(outputMCVar)})
#
#    del df, weight, y_hat, bin_index, outputWeighted, outputWeightedVar, outputMC, outputMCVar
#  
#  totalBkgOutput = np.array([b['outputScore'] for b in Background]) 
#  totalBkgOutput = totalBkgOutput.sum(axis=0)
#  
#  totalBkgVar = np.array([b['output_var'] for b in Background])
#  totalBkgVar = totalBkgVar.sum(axis=0)
#   
#  for s in Signal:
#    significance = []
#    significance_err = []
#    tot_rel = np.sqrt(np.sum(s['output_var'])) / s['nEvents']
#    for i in range(len(bins[1:])):
#      #eff_sig = s['outputScore'][:i+1].sum() / s['nEvents']
#      #eff_bkg = totalBkgOutput[:i+1].sum() / totalBkgOutput.sum()
#      eff_sig = s['outputScore'][i:-1].sum() / s['nEvents']
#      eff_bkg = totalBkgOutput[i:-1].sum() / totalBkgOutput.sum()
# 
#      #err_sig = np.sqrt(np.sum(s['output_var'][:i+1])) / s['nEvents']
#      #err_bkg = np.sqrt(np.sum(totalBkgVar[:i+1])) / totalBkgOutput.sum()
#      err_sig = np.sqrt(np.sum(s['output_var'][i:-1])) / s['nEvents']
#      err_bkg = np.sqrt(np.sum(totalBkgVar[i:-1])) / totalBkgOutput.sum()
#
#      #if totalBkgOutput[:i+1].sum() > 0.:
#      #  rel_err_bkg = np.sqrt(np.sum(totalBkgVar[:i+1])) / totalBkgOutput[:i+1].sum()
#      if totalBkgOutput[i:-1].sum() > 0.:
#        rel_err_bkg = np.sqrt(np.sum(totalBkgVar[i:-1])) / totalBkgOutput[i:-1].sum()
#      else:
#        rel_err_bkg = 0.
#      #if s['outputScore'][:i+1].sum() > 0.:
#      #  rel_err_sig = np.sqrt(np.sum(s['output_var'][:i+1])) / s['outputScore'][:i+1].sum()
#      if s['outputScore'][i:-1].sum() > 0.:
#        rel_err_sig = np.sqrt(np.sum(s['output_var'][i:-1])) / s['outputScore'][i:-1].sum()
#      else:
#        rel_err_sig = 0.
#      
#      total_rel_err = np.sqrt(rel_err_sig**2. + rel_err_bkg**2. + 0.25**2.)
#
#      if (eff_sig == 0) or (eff_bkg == 0):
#        Z = 0.
#        Z_err = 0.
#      elif (err_sig / eff_sig > 0.75) or (err_bkg / eff_bkg > 0.75):
#        Z = 0
#        Z_err = 0
#      else:
#        #Z = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(s['outputScore'][:i+1].sum(), totalBkgOutput[:i+1].sum(), total_rel_err)
#        Z = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(s['outputScore'][i:-1].sum(), totalBkgOutput[i:-1].sum(), total_rel_err)
#
#        Zplus_sig = ROOT.RooStats.NumberCountingUtils.BinomialExpZ((eff_sig + err_sig) * s['nEvents'], eff_bkg * totalBkgOutput.sum(), total_rel_err)
#        Zmins_sig = ROOT.RooStats.NumberCountingUtils.BinomialExpZ((eff_sig - err_sig) * s['nEvents'], eff_bkg * totalBkgOutput.sum(), total_rel_err)
#        Zplus_bkg = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(eff_sig * s['nEvents'], (eff_bkg + err_bkg) * totalBkgOutput.sum(), total_rel_err)
#        Zmins_bkg = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(eff_sig * s['nEvents'], (eff_bkg - err_bkg) * totalBkgOutput.sum(), total_rel_err)
#
#      Z_err_sig = abs(Zplus_sig - Zmins_sig) / 2
#      Z_err_bkg = abs(Zplus_bkg - Zmins_bkg) / 2
#      Z_err = np.sqrt(Z_err_sig**2 + Z_err_bkg**2)
#
#      significance.append(Z)
#      significance_err.append(Z_err)
#
#    s['sig'] = np.array(significance)
#    s['sig_max'] = s['sig'].max()
#    s['sig_err'] = np.array(significance_err)
#    print s['sig']
#    print s['sig'].max(), bins[np.where(s['sig'] == s['sig'].max())]
#
#  x = np.array([s['m_stop'] for s in Signal], dtype=float)
#  y = np.array([s['m_X'] for s in Signal], dtype=float)
#  z = np.array([s['sig_max'] for s in Signal],dtype=float)
#
#  print x, y, z
#  # Set up a regular grid of interpolation points
#  fig, ax1 = plt.subplots(figsize=(8,6))
#  xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
#  xi, yi = np.meshgrid(xi, yi)
#
#  # Interpolate
#  rbf = scipy.interpolate.LinearNDInterpolator(points=np.array((x, y)).T, values=z)
#  zi = rbf(xi, yi)
#
#  im = ax1.imshow(zi, vmin=0., vmax=5., origin='lower',
#             extent=[x.min(), x.max(), y.min(), y.max()])
#  cbar = plt.colorbar(im)
#  cbar.set_label('Significance')
#  ax1.set_xlabel(r'$m_{\tilde{t}}$')
#  ax1.set_xlim([x.min(), x.max()])
#  ax1.set_ylabel(r'$m_{\chi}$')
#  ax1.set_ylim([y.min(), y.max()])
#  plt.scatter(x, y, c='black')
#  plt.plot(x, x-84., color='black')
#  plt.plot(x, x-175., color='black')
#  AtlasStyle_mpl.ATLASLabel(ax1, 0.022, 0.925, 'Work in progress')
#  AtlasStyle_mpl.LumiLabel(ax1, 0.022, 0.875, lumi=lumi*0.001)
#  #plt.show()
#  
#  if save:
#        if not os.path.exists("./plots/"):
#            os.makedirs("./plots/")
#            print("Creating folder plots")
#        plt.savefig("plots/"+fileName+"_evaluated_grid.pdf")
#        plt.savefig("plots/"+fileName+"_evaluated_grid.png")
#        plt.close()
