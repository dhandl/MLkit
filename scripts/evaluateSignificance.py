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
from prepareSequentialTraining import loadDataFrame as loadSequentialDataFrame
from getRatio import getRatio

from collections import namedtuple
Sample = namedtuple('Sample', 'name' 'path')

sys.path.append('./python/plotting/')
import plot_ROCcurves

#inputDir = '/project/etp5/dhandl/samples/SUSY/Stop1L/hdf5/cut_mt30_met60_preselection/'
inputDir = '/project/etp5/dhandl/samples/SUSY/Stop1L/FullRun2/hdf5/cut_mt30_met60_preselection/'

#SIGNAL = ['stop_bWN_250_100', 'stop_bWN_250_130', 'stop_bWN_300_150', 'stop_bWN_300_180', 'stop_bWN_350_200', 'stop_bWN_350_230', 'stop_bWN_400_250', 'stop_bWN_400_280', 'stop_bWN_450_300', 'stop_bWN_450_330', 'stop_bWN_500_350', 'stop_bWN_500_380', 'stop_bWN_550_400', 'stop_bWN_550_430', 'stop_bWN_600_450', 'stop_bWN_600_480', 'stop_bWN_650_500', 'stop_bWN_650_530']
SIGNAL = ['stop_bWN_450_300_mc16d']
#SIGNAL = ['stop_tN_800_500']

BACKGROUND = ['mc16e_ttbar', 'mc16e_singletop', 'mc16e_wjets']

PRESELECTION = [
                {'name':'n_jet',  'threshold':4,      'type':'geq'},
                {'name':'n_bjet',  'threshold':1,      'type':'geq'},
                {'name':'met',    'threshold':230e3,  'type':'geq'},
                {'name':'mt',    'threshold':150e3,  'type':'geq'},
                {'name':'n_lep',  'threshold':1,      'type':'exact'},
                {'name':'lep_pt',  'threshold':25e3,      'type':'geq'}
               ]

WEIGHTS = [
           'weight',
           'xs_weight',
           'sf_total'
          ]

LUMI = 140e3

RESOLUTION = np.array([50,0,1], dtype=float)

COLLECTION = ['jet'] 
REMOVE_VAR = ['_m', '_mv2c10', '_id', '0_pt', '0_eta', '0_phi', '0_e', '1_pt', '1_eta', '1_phi', '1_e']


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


def evaluate(model, dataset, scaler, seq_scaler=None, col=None, rnn=False):

  #where_nan = np.isnan(dataset)
  #dataset[where_nan] = -999. 
  dataset = scaler.transform(dataset)

  if rnn:  
    for idx, c in enumerate(col):
      #c['n_max'] = max([len(j) for j in c['df'][c['name']+'_pt']])
      c['n_max'] = 15
      c['Xobj'] = create_scale_stream(c['df'], c['n_max'], sort_col=c['name']+'_pt', VAR_FILE_NAME=seq_scaler) 

    y_hat = model.predict([c['Xobj'] for c in col]+[dataset])

  else:
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


def create_scale_stream(df, num_obj, sort_col, VAR_FILE_NAME):
  n_variables = df.shape[1]
  var_names = df.keys()
  data = np.zeros((df.shape[0], num_obj, n_variables), dtype='float64')
  
  # call functions to build X (a.k.a. data)
  sort_objects(df, data, sort_col, num_obj)
  Xobj = data
  
  scale(Xobj, var_names, VAR_FILE_NAME=VAR_FILE_NAME.replace('.h5','_scaling.json')) # apply scaling
  return Xobj


def sort_objects(df, data, SORT_COL, max_nobj):
  ''' 
  sort objects using your preferred variable
  Args:
  -----
      df: a dataframe with event-level structure where each event is described by a sequence of jets, muons, etc.
      data: an array of shape (nb_events, nb_particles, nb_features)
      SORT_COL: a string representing the column to sort the objects by
      max_nobj: number of particles to cut off at. if >, truncate, else, -999 pad
  
  Returns:
  --------
      modifies @a data in place. Pads with -999
  
  '''
  import tqdm
  # i = event number, event = all the variables for that event
  idx = 0 
  for i, event in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
    # objs = [[pt's], [eta's], ...] of particles for each event

    #objs = np.array([v.tolist() for v in event.get_values()], dtype='float32')[:, (np.argsort(event[SORT_COL]))[::-1]]
    objs = np.array([v.tolist() for v in event.get_values()], dtype='float64')[:, (np.argsort(event[SORT_COL]))[::-1]]
    # total number of tracks per jet      
    nobjs = objs.shape[1] 
    # take all tracks unless there are more than n_tracks 
    data[idx, :(min(nobjs, max_nobj)), :] = objs.T[:(min(nobjs, max_nobj)), :] 
    # default value for missing tracks 
    data[idx, (min(nobjs, max_nobj)):, :  ] = -999
    idx = idx + 1

def scale(data, var_names, VAR_FILE_NAME):
  import json
  scale = {}
  with open(VAR_FILE_NAME, 'rb') as varfile:
    varinfo = json.load(varfile)

  for v, name in enumerate(var_names):
    #print 'Scaling feature %s of %s (%s).' % (v + 1, len(var_names), name)
    f = data[:, :, v]
    slc = f[f != -999]
    m = varinfo[name]['mean']
    s = varinfo[name]['sd']
    slc -= m
    slc /= s
    data[:, :, v][f != -999] = slc.astype('float64')


def main():

  # Check number of arguments and act respectively thereof
  if len(sys.argv) == 2:
    modelfile = sys.argv[1:][0]
  else:
    print 'Usage: evaluate_signal.py <model> (omit directory and file suffix)'
    return

  print modelfile, type(modelfile)

  Dir = 'TrainedModels/models/'
  DatasetDir = 'TrainedModels/datasets/'
  
  modelDir = Dir+modelfile+'.h5'

  model = load_model(os.path.join(Dir, modelfile+'.h5'))

  scaler = joblib.load(os.path.join(Dir,modelfile+'_scaler.pkl'))

  infofile = open(modelDir.replace('.h5','_infofile.txt'))
  infos = infofile.readlines()
  analysis=infos[0].replace('Used analysis method: ','').replace('\n','')
  dataset = DatasetDir + infos[3].replace('Used dataset: ', '').replace('\n','')
  VAR = infos[5].replace('Used variables for training: ', '').replace('\n','').split()
  print VAR
  recurrent = False
  if analysis.lower() == 'rnn':
    recurrent = True
    seq_scaler = dataset+'_scaling.json'

  db = (RESOLUTION[2] - RESOLUTION[1]) / RESOLUTION[0]    # bin width in discriminator distribution
  bins = np.arange(RESOLUTION[1], RESOLUTION[2]+db, db)   # bin edges in discriminator distribution
  center = (bins[:-1] + bins[1:]) / 2

  print '#----MODEL----#'
  print modelDir

  ###########################
  # Read and evaluate signals
  ###########################

  Signal = []
  for s in SIGNAL:
    x, y = pickBenchmark(s)
    if not recurrent:
      df, weight = loadDataFrame(os.path.join(inputDir, s+'/'), PRESELECTION, VAR, WEIGHTS, LUMI)
      y_hat = evaluate(model, df.values, scaler)
    else: 
      df, weight, collection = loadSequentialDataFrame(os.path.join(inputDir, s+'/'), PRESELECTION, COLLECTION, REMOVE_VAR, VAR, WEIGHTS, LUMI)
      y_hat = evaluate(model, df.values, scaler, seq_scaler, rnn=True, col=collection)

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
    if not recurrent:
      df, weight = loadDataFrame(os.path.join(inputDir, b+'/'), PRESELECTION, VAR, WEIGHTS, LUMI)
      y_hat = evaluate(model, df.values, scaler)
    else:
      df, weight, collection = loadSequentialDataFrame(os.path.join(inputDir, b+'/'), PRESELECTION, COLLECTION, REMOVE_VAR, VAR, WEIGHTS, LUMI)
      y_hat = evaluate(model, df.values, scaler, seq_scaler, rnn=True, col=collection)

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
    asimov_err = []
    roc = []
    roc_err = []

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
        ams_err = 0.
      elif (err_sig / eff_sig > 0.75) or (err_bkg / eff_bkg > 0.75):
        Z = 0.
        Z_err = 0.
        ams = 0.
        ams_err = 0.
      else:
        #Z = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(s['outputScore'][:i+1].sum(), totalBkgOutput[:i+1].sum(), total_rel_err)
        Z = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(s['outputScore'][i:].sum(), totalBkgOutput[i:].sum(), total_rel_err)
        ams = asimovZ( s['outputScore'][i:].sum(), totalBkgOutput[i:].sum(), np.sqrt(totalBkgVar[i:].sum()))
        roc.append((eff_sig, 1-eff_bkg))

        ams_plus_sig = asimovZ((s['outputScore'][i:].sum() + np.sqrt(np.sum(s['output_var'][i:]))), totalBkgOutput[i:].sum(), np.sqrt(totalBkgVar[i:].sum()))
        ams_mins_sig = asimovZ((s['outputScore'][i:].sum() - np.sqrt(np.sum(s['output_var'][i:]))), totalBkgOutput[i:].sum(), np.sqrt(totalBkgVar[i:].sum()))
        ams_plus_bkg = asimovZ(s['outputScore'][i:].sum(), (totalBkgOutput[i:].sum() + np.sqrt(np.sum(totalBkgVar[i:]))), np.sqrt(totalBkgVar[i:].sum()))
        ams_mins_bkg = asimovZ(s['outputScore'][i:].sum(), (totalBkgOutput[i:].sum() - np.sqrt(np.sum(totalBkgVar[i:]))), np.sqrt(totalBkgVar[i:].sum()))

        Zplus_sig = ROOT.RooStats.NumberCountingUtils.BinomialExpZ((eff_sig + err_sig) * s['nEvents'], eff_bkg * totalBkgOutput.sum(), total_rel_err)
        Zmins_sig = ROOT.RooStats.NumberCountingUtils.BinomialExpZ((eff_sig - err_sig) * s['nEvents'], eff_bkg * totalBkgOutput.sum(), total_rel_err)
        Zplus_bkg = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(eff_sig * s['nEvents'], (eff_bkg + err_bkg) * totalBkgOutput.sum(), total_rel_err)
        Zmins_bkg = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(eff_sig * s['nEvents'], (eff_bkg - err_bkg) * totalBkgOutput.sum(), total_rel_err)

        Z_err_sig = abs(Zplus_sig - Zmins_sig) / 2
        Z_err_bkg = abs(Zplus_bkg - Zmins_bkg) / 2
        Z_err = np.sqrt(Z_err_sig**2 + Z_err_bkg**2)

        ams_err_sig = abs(ams_plus_sig - ams_mins_sig)/2.
        ams_err_bkg = abs(ams_plus_bkg - ams_mins_bkg)/2.
        ams_err = np.sqrt(ams_err_sig**2 + ams_err_bkg**2)

      significance.append(Z)
      significance_err.append(Z_err)
      asimov.append(ams)
      asimov_err.append(ams_err)

    s['sig'] = np.array(significance)
    s['sig_max'] = s['sig'].max()
    s['sig_err'] = np.array(significance_err)
    s['ams'] = np.array(asimov)
    s['ams_err'] = np.array(asimov_err)
    s['roc'] = np.array(roc)

    print s['sig']
    print s['ams']
    #print s['roc']
    sigMax_index = bins[np.where(s['sig'] == s['sig'].max())][0]
    amsMax_index = bins[np.where(s['ams'] == s['ams'].max())][0]
    Z = asimovZ(Signal[0]['outputScore'][np.where(bins[:-1] == sigMax_index)], totalBkgOutput[np.where(bins[:-1] == sigMax_index)], np.sqrt(totalBkgVar[np.where(bins[:-1] == sigMax_index)]), syst=False)
    Z_syst = asimovZ(Signal[0]['outputScore'][np.where(bins[:-1] == sigMax_index)], totalBkgOutput[np.where(bins[:-1] == sigMax_index)], np.sqrt(totalBkgVar[np.where(bins[:-1] == sigMax_index)]), syst=True)
    print 'RooStats: ',s['sig'].max(), sigMax_index, Z, Z_syst
    print 'asmiov : ', s['ams'].max(), amsMax_index

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
  ax1.set_xlabel('Output score', horizontalalignment='right', x=1.0)
  ax1.set_ylabel("Events", horizontalalignment='right', y=1.0)

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

  w = plt.bar(center, Background[2]['outputScore'], width=db, yerr=np.sqrt(Background[2]['output_var']), color='gold', alpha=0.5, error_kw=dict(ecolor='gold', lw=1.5), label='W+jets')  
  st = plt.bar(center, Background[1]['outputScore'], width=db, yerr=np.sqrt(Background[1]['output_var']), color='limegreen', alpha=0.5, error_kw=dict(ecolor='limegreen', lw=1.5), label='singletop', bottom=Background[2]['outputScore'])  
  tt = plt.bar(center, Background[0]['outputScore'], width=db, yerr=np.sqrt(Background[0]['output_var']), color='dodgerblue', alpha=0.5, error_kw=dict(ecolor='dodgerblue', lw=1.5), label='ttbar', bottom=Background[2]['outputScore']+Background[1]['outputScore'])  
  plt.bar(center, Signal[0]['outputScore'], width=db, yerr= np.sqrt(Signal[0]['output_var']), label=Signal[0]['name'], color='r', alpha=0.5, error_kw=dict(ecolor='r', lw=1.5))  

  ax1.set_ylim((0.1, totalBkgOutput.max()*(15.)))
  ax1.set_yscale('log')
  leg = plt.legend(loc="best", frameon=False)

  AtlasStyle_mpl.ATLASLabel(ax1, 0.14, 0.85, 'Work in progress')
  AtlasStyle_mpl.LumiLabel(ax1, 0.02, 0.87, lumi=LUMI*0.001)

  plt.savefig("plots/"+modelfile+"_eval-bWN-450-300_outputScore.pdf")
  plt.savefig("plots/"+modelfile+"_eval-bWN-450-300_outputScore.png")
  plt.close()

  print('Plotting significance...')
  fig = plt.figure(figsize=(8,6))
  ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=4)
  ax1.set_xlim((bins[0], bins[-1]))
  ax1.set_xlabel('Output score', horizontalalignment='right', x=1.0)
  ax1.set_ylabel("Z", horizontalalignment='right', y=1.0)

  plt.plot(center, Signal[0]['ams'], 'k-', color='cornflowerblue', label='Asimov Z (max = %0.3f at %0.2f)'%(s['ams'].max(), amsMax_index))
  plt.fill_between(center, Signal[0]['ams']-Signal[0]['ams_err'], Signal[0]['ams']+Signal[0]['ams_err'], alpha=0.2, edgecolor='cornflowerblue', facecolor='cornflowerblue', linewidth=0)
  ax1.set_ylim((0., Signal[0]['ams'].max()*(1.5)))

  plt.plot(center, Signal[0]['sig'], 'k-', color='darkred', label='Binomial Z (max = %0.3f at %0.2f)'%(s['sig'].max(), sigMax_index))
  plt.fill_between(center, Signal[0]['sig']-Signal[0]['sig_err'], Signal[0]['sig']+Signal[0]['sig_err'], alpha=0.2, edgecolor='darkred', facecolor='darkred', linewidth=0)
  plt.plot(center, len(center)*[3.], '--', color='grey', alpha=0.5)
  plt.plot(center, len(center)*[5.], '--', color='red', alpha=0.5)
  leg = plt.legend(loc="best", frameon=False)

  AtlasStyle_mpl.ATLASLabel(ax1, 0.14, 0.85, 'Work in progress')
  AtlasStyle_mpl.LumiLabel(ax1, 0.02, 0.87, lumi=LUMI*0.001)

  plt.savefig("plots/"+modelfile+"_Significance_bWN-450-300.pdf")
  plt.savefig("plots/"+modelfile+"_Significance_bWN-450-300.png")
  plt.close()

  print('Plotting ROC...')
  fig = plt.figure(figsize=(8,6))
  ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=4)
  ax1.set_xlim((bins[0], bins[-1]))
  ax1.set_ylim((0, 1))
  ax1.set_xlabel('$\epsilon_{Sig.}$', horizontalalignment='right', x=1.0)
  ax1.set_ylabel("$r_{Bkg.}$", horizontalalignment='right', y=1.0)

  auc = np.trapz(s['roc'][:,0], s['roc'][:,1], dx=db)
  print 'Area under ROC?!: ',auc

  plt.plot(s['roc'][:,0], s['roc'][:,1], 'k-', color='cornflowerblue', label='ROC (AUC = %0.4f)'%(auc))
  #plt.fill_between(center, Signal[0]['ams']-Signal[0]['ams_err'], Signal[0]['ams']+Signal[0]['ams_err'], alpha=0.2, edgecolor='cornflowerblue', facecolor='cornflowerblue', linewidth=0)
  plt.plot([0, 1], [1, 0], '--', color=(0.6, 0.6, 0.6), label='Luck')
  leg = plt.legend(loc="best", frameon=False)

  AtlasStyle_mpl.ATLASLabel(ax1, 0.14, 0.2, 'Work in progress')
  AtlasStyle_mpl.LumiLabel(ax1, 0.02, 0.05, lumi=LUMI*0.001)

  plt.savefig("plots/"+modelfile+"_ROC_bWN-450-300.pdf")
  plt.savefig("plots/"+modelfile+"_ROC_bWN-450-300.png")
  plt.close()

if __name__ == "__main__":
    main()
   

