#!/usr/bin/env python

import os, sys

import pandas as pd
import numpy as np

from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import ROOT
ROOT.gSystem.Load('libRooStats')

from prepareTraining import loadDataFrame, weightFrame, selectVarList, applyCut, varHasIndex, pickIndex
from getRatio import getRatio

from collections import namedtuple
Sample = namedtuple('Sample', 'name' 'path')


inputDir = os.path.join(os.getenv('SampleDir'), 'stop1l/hdf5/')

#Dir = 'TrainedModels/models/'
#modelfile = '2018-05-11_14-54_DNN_rmsprop_layer256_batch16_GlorotUniformInitializer_dropout0p5_l1-0p01'

#modelDir = Dir+modelfile+'.h5'

#SIGNAL = ['stop_tN_500_327']
#SIGNAL = ['stop_tN_800_500']

BACKGROUND = ['bkgs']

PRESELECTION = [
                {'name':'n_jet',  'threshold':4,      'type':'geq'},
                {'name':'n_bjet',  'threshold':1,      'type':'geq'},
                {'name':'met',    'threshold':250e3,  'type':'geq'},
                {'name':'mt',    'threshold':110e3,  'type':'geq'},
                {'name':'n_lep',  'threshold':1,      'type':'exact'}
               ]

VAR = [
        'met',
        'met_phi',
        'n_jet',
        'lep_pt[0]',
        'lep_eta[0]',
        'lep_phi[0]',
        'jet_pt[0]',
        'jet_eta[0]',
        'jet_phi[0]',
        'jet_m[0]',
        'jet_bweight[0]',
        'jet_pt[1]',
        'jet_eta[1]',
        'jet_phi[1]',
        'jet_m[1]',
        'jet_bweight[1]',
        'jet_pt[2]',
        'jet_eta[2]',
        'jet_phi[2]',
        'jet_m[2]',
        'jet_bweight[2]',
        'jet_pt[3]',
        'jet_eta[3]',
        'jet_phi[3]',
        'jet_m[3]',
        'jet_bweight[3]'
      ]

WEIGHTS = [
           'event_weight'
          ]

LUMI = 1.

RESOLUTION = np.array([50,0,1], dtype=float)

#SCALING = Dir+modelfile+'_scaler.pkl'

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


def parse_options():
  import argparse

  workdir = os.getenv('MLDir')
  output = os.path.join(workdir, 'TrainedModels')

  parser = argparse.ArgumentParser()
  parser.add_argument('-n', '--name', help='Name of the model')
  parser.add_argument('-s', '--signal', help='Name of the signal sample (i.e. "stop_tN_500_327")', type=str)

  opts = parser.parse_args()

  opts.weightDir = os.path.join(output, 'weights')
  opts.modelDir = os.path.join(output, 'models')

  return opts


def main():
  
  opts = parse_options()
  
  model = load_model(os.path.join(opts.modelDir, opts.name+'.h5'))
  
  scaler = joblib.load(os.path.join(opts.modelDir, opts.name+'_scaler.pkl'))

  db = (RESOLUTION[2] - RESOLUTION[1]) / RESOLUTION[0]    # bin width in discriminator distribution
  bins = np.arange(RESOLUTION[1], RESOLUTION[2]+db, db)   # bin edges in discriminator distribution
  center = (bins[:-1] + bins[1:]) / 2


  ###########################
  # Read and evaluate signals
  ###########################


  SIGNAL = [opts.signal]
  Signal = []
  for s in SIGNAL:
    x, y = pickBenchmark(s)
    df, weight = loadDataFrame(os.path.join(inputDir, s), PRESELECTION, VAR, WEIGHTS, LUMI)
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
    df, weight = loadDataFrame(os.path.join(inputDir, b), PRESELECTION, VAR, WEIGHTS, LUMI)
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
  
  
  ###########################
  # Determine Significance  #
  ###########################
 

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
    #print s['sig']
    #print s['ams']
    sigMax_index = bins[np.where(s['sig'] == s['sig'].max())][0]
    Z = asimovZ(Signal[0]['outputScore'][np.where(bins[:-1] == sigMax_index)], totalBkgOutput[np.where(bins[:-1] == sigMax_index)], np.sqrt(totalBkgVar[np.where(bins[:-1] == sigMax_index)]), syst=False)
    Z_syst = asimovZ(Signal[0]['outputScore'][np.where(bins[:-1] == sigMax_index)], totalBkgOutput[np.where(bins[:-1] == sigMax_index)], np.sqrt(totalBkgVar[np.where(bins[:-1] == sigMax_index)]), syst=True)
    #print s['sig'].max(), sigMax_index, Z, Z_syst

  x = np.array([s['m_stop'] for s in Signal], dtype=float)
  y = np.array([s['m_X'] for s in Signal], dtype=float)
  z = np.array([s['sig_max'] for s in Signal],dtype=float)

  #print x, y, z

  #print Signal[0]['outputScore'][np.where(bins[:-1] >= sigMax_index)], Signal[0]['output_var'][np.where(bins[:-1] >= sigMax_index)]
  #print totalBkgOutput[np.where(bins[:-1] >= sigMax_index)], totalBkgVar[np.where(bins[:-1] >= sigMax_index)]

  #print Signal[0]['outputScore'], Signal[0]['output_var']
  #print totalBkgOutput, totalBkgVar

  
  ###################################
  # Write single bin to .root files #
  ##################################


  sigFile = ROOT.TFile(opts.name+"_output_sig.root", "RECREATE")
  sig_sr = ROOT.TH1D("SR", "SR", 1,0,1)
  sig_sr.SetBinContent(1, np.sum(Signal[0]['outputScore'][np.where(bins[:-1] >= sigMax_index)]))
  sig_sr.SetBinError(1, np.sum(np.sqrt(Signal[0]['output_var'][np.where(bins[:-1] >= sigMax_index)])))
  sigFile.Write()
  sigFile.Close()

  bkgFile = ROOT.TFile(opts.name+"_output_bkg.root", "RECREATE")
  bkg_sr = ROOT.TH1D("SR", "SR", 1,0,1)
  bkg_sr.SetBinContent(1, np.sum(totalBkgOutput[np.where(bins[:-1] >= sigMax_index)]))
  bkg_sr.SetBinError(1, np.sum(np.sqrt(totalBkgVar[np.where(bins[:-1] >= sigMax_index)])))
  bkgFile.Write()
  bkgFile.Close()
  

  ###################################
  # Write multi bins to .root files #
  ###################################
  
  
  multibin_sigFile = ROOT.TFile(opts.name+"_output_sig_multibin.root", "RECREATE")
  multibin_sig_sr = ROOT.TH1D("SR", "SR", 5,0,5)
  for i in xrange(1,6):
    index = -6 + i
    multibin_sig_sr.SetBinContent(i, Signal[0]['outputScore'][index])
    multibin_sig_sr.SetBinError(i, np.sqrt(Signal[0]['output_var'][index]))
  multibin_sigFile.Write()
  multibin_sigFile.Close()

  multibin_bkgFile = ROOT.TFile(opts.name+"_output_bkg_multibin.root", "RECREATE")
  multibin_bkg_sr = ROOT.TH1D("SR", "SR", 5,0,5)
  for i in xrange(1,6):
    index = -6 + i
    multibin_bkg_sr.SetBinContent(i, totalBkgOutput[index])
    multibin_bkg_sr.SetBinError(i, np.sqrt(totalBkgVar[index]))
  multibin_bkgFile.Write()
  multibin_bkgFile.Close()
 
 
if __name__ == "__main__":
    main()

