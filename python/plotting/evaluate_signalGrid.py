#!/usr/bin/env python

import os, sys

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate

import AtlasStyle_mpl

from keras.models import load_model
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import ROOT
ROOT.gSystem.Load('libRooStats')

from prepareTraining import loadDataFrame, weightFrame, selectVarList, applyCut, varHasIndex, pickIndex
from prepareSequentialTraining import loadDataFrame as loadSequentialDataFrame

import matplotlib.patches as mpatches

from collections import namedtuple
Sample = namedtuple('Sample', 'name' 'path')

inputDirSig = '/project/etp5/dhandl/samples/SUSY/Stop1L/FullRun2/hdf5/MC15_signals/'
inputDirBkg = '/project/etp5/dhandl/samples/SUSY/Stop1L/FullRun2/hdf5/cut_mt30_met60_preselection_new/'

Dir = 'TrainedModels/models/'
DatasetDir = 'TrainedModels/datasets/'

modelfile = '2018-09-28_00-38_RNN_jetOnly_ADAM_leayReLU_LSTM32_128NNlayer_batch32_NormalInitializer_l2-0p01'
#modelfile = '2018-08-02_15-24_DNN_ADAM_layer1x100_batch10_NormalInitializer_dropout0p5_l1-0p01'
#modelfile = '2018-08-02_16-03_DNN_ADAM_layer1x100_batch10_NormalInitializer_dropout0p5_l1-0p01'
#modelfile = '2018-07-31_13-16_DNN_ADAM_layer1x100_batch40_NormalInitializer_dropout0p5_l1-0p01_multiclass'
#modelfile = '2018-07-31_14-15_DNN_ADAM_layer1x100_batch40_NormalInitializer_dropout0p5_l1-0p01_multiclass'

modelDir = Dir+modelfile+'.h5'
SIGNAL = ['stop_bWN_190_23', 'stop_bWN_220_53', 'stop_bWN_250_100', 'stop_bWN_250_130', 'stop_bWN_250_160', 'stop_bWN_300_150', 'stop_bWN_300_180', 'stop_bWN_300_210', 'stop_bWN_350_185', 'stop_bWN_350_200', 'stop_bWN_350_230', 'stop_bWN_350_260', 'stop_bWN_400_235', 'stop_bWN_400_250', 'stop_bWN_400_280', 'stop_bWN_400_310', 'stop_bWN_450_285', 'stop_bWN_450_300', 'stop_bWN_450_330', 'stop_bWN_450_360', 'stop_bWN_500_335', 'stop_bWN_500_350', 'stop_bWN_500_380', 'stop_bWN_550_385', 'stop_bWN_550_400', 'stop_bWN_550_430', 'stop_bWN_550_460', 'stop_bWN_600_435', 'stop_bWN_600_450', 'stop_bWN_600_480', 'stop_bWN_600_510', 'stop_bWN_650_485', 'stop_bWN_650_500', 'stop_bWN_650_530', 'stop_bWN_650_560', 'stop_tN_190_17', 'stop_tN_250_62', 'stop_tN_250_77', 'stop_tN_195_1', 'stop_tN_200_12', 'stop_tN_200_27', 'stop_tN_350_162', 'stop_tN_350_177', 'stop_tN_300_112', 'stop_tN_300_127', 'stop_tN_350_150', 'stop_tN_450_250', 'stop_tN_450_262', 'stop_tN_400_200', 'stop_tN_400_212', 'stop_tN_400_227', 'stop_tN_500_327', 'stop_tN_550_350', 'stop_tN_450_277', 'stop_tN_500_300', 'stop_tN_500_312', 'stop_tN_600_412', 'stop_tN_600_427', 'stop_tN_550_362', 'stop_tN_550_377', 'stop_tN_600_400','stop_tN_650_450', 'stop_tN_650_462', 'stop_tN_650_477']
#SIGNAL = [ 'stop_bWN_190_23', 'stop_bWN_220_53', 'stop_bWN_250_100', 'stop_bWN_250_130', 'stop_bWN_250_160', 'stop_bWN_300_150', 'stop_bWN_300_180', 'stop_bWN_300_210', 'stop_bWN_350_185', 'stop_bWN_350_200', 'stop_bWN_350_230', 'stop_bWN_350_260', 'stop_bWN_400_250', 'stop_bWN_400_280', 'stop_bWN_450_300', 'stop_bWN_450_330', 'stop_bWN_500_350', 'stop_bWN_500_380', 'stop_bWN_550_400', 'stop_bWN_550_430', 'stop_bWN_600_450', 'stop_bWN_600_480', 'stop_bWN_650_500', 'stop_bWN_650_530']

BACKGROUND = ['mc16d_ttbar', 'mc16d_singletop', 'mc16d_wjets']

PRESELECTION = [
                {'name':'n_jet',  'threshold':4,      'type':'geq'},
                {'name':'n_bjet',  'threshold':1,      'type':'geq'},
                {'name':'met',    'threshold':230e3,  'type':'geq'},
                {'name':'mt',    'threshold':110e3,  'type':'geq'},
                {'name':'n_lep',  'threshold':1,      'type':'exact'},
                {'name':'lep_pt',  'threshold':25e3,      'type':'geq'}
               ]

VAR = [
#        'jet_pt[0]',
#        'bjet_pt[0]',
#        'amt2',
#        'mt',
#        'met',
#        'dphi_met_lep',
#        #'met_phi',
##        #'dphi_b_lep_max',
##        #'dphi_jet0_ptmiss',
#        'met_proj_lep',
#        'ht_sig',
#        'm_bl',
#        'lepPt_over_met'
#        #'dr_bjet_lep',
#        #'mT_blMET', #15vars
        #'n_jet',
        #'n_bjet',
         #'bjet_pt[0]', 'amt2', 'mt', 'met', 'dphi_met_lep', 'ht_sig', 'dr_bjet_lep'
        #'met',
        #'mt',
        #'Lp',
        #'dphi_met_lep',
        #'met_sig',
        #'m3'
        'met',
        'met_phi',
        'dphi_met_lep',
        'mt',
        'n_jet',
        'n_bjet',
        #'jet_pt[0]',
        #'jet_eta[0]',
        #'jet_phi[0]',
        #'jet_e[0]',
        ##'jet_bweight[0]',
        #'jet_pt[1]',
        #'jet_eta[1]',
        #'jet_phi[1]',
        #'jet_e[1]',
        ##'jet_bweight[1]',
        #'jet_pt[2]',
        #'jet_eta[2]',
        #'jet_phi[2]',
        #'jet_e[2]',
        ##'jet_bweight[2]',
        #'jet_pt[3]',
        #'jet_eta[3]',
        #'jet_phi[3]',
        #'jet_e[3]',
        ##'jet_bweight[3]'
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
  data = np.zeros((df.shape[0], num_obj, n_variables), dtype='float32')
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
  i = 0 
  for idx, event in tqdm.tqdm(df.iterrows(), total=df.shape[0]): 
    # objs = [[pt's], [eta's], ...] of particles for each event

    #objs = np.array([v.tolist() for v in event.get_values()], dtype='float32')[:, (np.argsort(event[SORT_COL]))[::-1]]
    objs = np.array([v for v in event.get_values()], dtype='float32')[:, (np.argsort(event[SORT_COL]))[::-1]]
    # total number of tracks per jet      
    nobjs = objs.shape[1]
    #print i, idx, event
    #print nobjs 
    #print objs
    # take all tracks unless there are more than n_tracks 
    data[i, :(min(nobjs, max_nobj)), :] = objs.T[:(min(nobjs, max_nobj)), :] 
    # default value for missing tracks 
    data[i, (min(nobjs, max_nobj)):, :  ] = -999
    i = i + 1

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
    data[:, :, v][f != -999] = slc.astype('float32')


def main():
  
  model = load_model(modelDir)

  scaler = joblib.load(SCALING)

  infofile = open(modelDir.replace('.h5','_infofile.txt'))
  infos = infofile.readlines()
  analysis=infos[0].replace('Used analysis method: ','').replace('\n','')
  dataset = DatasetDir + infos[3].replace('Used dataset: ', '').replace('\n','')
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
    print s
    x, y = pickBenchmark(s)
    if not recurrent:
      df, weight = loadDataFrame(os.path.join(inputDirSig, s+'/'), PRESELECTION, VAR, WEIGHTS, LUMI)
      y_hat = evaluate(model, df.values, scaler)
    else:
      df, weight, collection = loadSequentialDataFrame(os.path.join(inputDirSig, s+'/'), PRESELECTION, COLLECTION, REMOVE_VAR, VAR, WEIGHTS, LUMI)
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
      df, weight = loadDataFrame(os.path.join(inputDirBkg, b+'/'), PRESELECTION, VAR, WEIGHTS, LUMI)
      y_hat = evaluate(model, df.values, scaler)
    else:
      df, weight, collection = loadSequentialDataFrame(os.path.join(inputDirBkg, b+'/'), PRESELECTION, COLLECTION, REMOVE_VAR, VAR, WEIGHTS, LUMI)
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
        Z = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(s['outputScore'][i:-1].sum(), totalBkgOutput[i:-1].sum(), total_rel_err)
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
    print s['m_stop'], s['m_X'], s['sig'].max(), bins[np.where(s['sig'] == s['sig'].max())]

  x = np.array([s['m_stop'] for s in Signal], dtype=float)
  y = np.array([s['m_X'] for s in Signal], dtype=float)
  z = np.array([s['sig_max'] for s in Signal],dtype=float)

  #print x, y, z
  # Set up a regular grid of interpolation points
  fig, ax1 = plt.subplots(figsize=(8,6))
  xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
  xi, yi = np.meshgrid(xi, yi)

  # Interpolate
  rbf = scipy.interpolate.LinearNDInterpolator(points=np.array((x, y)).T, values=z)
  zi = rbf(xi, yi)

  im = ax1.imshow(zi, vmin=0., vmax=5., origin='lower',
             extent=[x.min(), x.max(), y.min(), y.max()])
  
  contours = plt.contour(xi, yi, zi, colors='black', levels=[3.])
  cbar = plt.colorbar(im)
  cbar.set_label('Significance')
  ax1.set_xlabel(r'$m_{\tilde{t}}$')
  ax1.set_xlim([x.min(), x.max()])
  ax1.set_ylabel(r'$m_{\chi}$')
  ax1.set_ylim([y.min(), y.max()])
  plt.scatter(x, y, c='black', s=[0.75]*len(x))
  plt.plot(x, x-84., color='grey')
  plt.plot(x, x-175., color='grey')

  AtlasStyle_mpl.ATLASLabel(ax1, 0.022, 0.925, 'Work in progress')
  AtlasStyle_mpl.LumiLabel(ax1, 0.022, 0.875, lumi=LUMI*0.001)

  plt.savefig("plots/"+modelfile+"_eval-Grid.pdf")
  plt.savefig("plots/"+modelfile+"_eval-Grid.png")
  plt.close()

if __name__ == '__main__':
    main()
    
##-------------------------------------------------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------------------------------------------------##
    
def evaluate_signalGrid(modelDir, resolution=np.array([50,0,1], dtype=float), save=False, fileName='Test'):
  print('Evaluating signal grid...')  
  
  infofile = open(modelDir.replace('.h5','_infofile.txt'))
  infos = infofile.readlines()
  
  #Parse Strings for correct datatypes
  
  variables=infos[4].replace('Used variables for training: ','').replace('\n','').split()
  weights=infos[5].replace('Used weights: ', '').replace('\n','').split()
  preselection_raw=infos[6].replace('Used preselection: ', '').replace('; \n', '').split(';')
  preselection=[]
  for x in preselection_raw:
      xdict = {}
      xdict['name']= x.split()[0].split('-')[0]
      xdict['threshold']= float(x.split()[1])
      xdict['type'] = x.split()[3]
      if xdict['type'] == 'condition':
          xdict['variable'] = x.split()[5]
          xdict['lessthan'] = float(x.split()[7])
          xdict['morethan'] = float(x.split()[10])
      preselection.append(xdict)
  lumi=float(infos[7].replace('Used Lumi: ','').replace('\n',''))
  background=infos[9].replace('Used background files: ','').replace('; \n','').replace(' ','').split(';')
  #signal=infos[8].replace('Used signal files: ','').replace('; \n','').replace(' ','').split(';')
  
  signal = ['stop_bWN_250_100', 'stop_bWN_250_130', 'stop_bWN_250_160', 'stop_bWN_300_150', 'stop_bWN_300_180', 'stop_bWN_300_210', 'stop_bWN_350_185', 'stop_bWN_350_200', 'stop_bWN_350_230', 'stop_bWN_350_260', 'stop_bWN_400_235', 'stop_bWN_400_250', 'stop_bWN_400_280', 'stop_bWN_400_310', 'stop_bWN_450_285', 'stop_bWN_450_300', 'stop_bWN_450_330', 'stop_bWN_450_360', 'stop_bWN_500_335', 'stop_bWN_500_350', 'stop_bWN_500_380', 'stop_bWN_550_385', 'stop_bWN_550_400', 'stop_bWN_550_430', 'stop_bWN_550_460', 'stop_bWN_600_435', 'stop_bWN_600_450', 'stop_bWN_600_480', 'stop_bWN_600_510', 'stop_bWN_650_485', 'stop_bWN_650_500', 'stop_bWN_650_530', 'stop_bWN_650_560']
  
   
  #For Debugging
  #print variables, type(variables)
  #print weights, type(variables)
  #print preselection, type(preselection[1])
  #print lumi, type(lumi)
  #print signal, type(signal)
  #print background, type(background)
   
  #Get Scaler and model from modelDir
   
  model = load_model(modelDir)
  
  scalerDir=modelDir.replace('.h5','_scaler.pkl')
  scaler=joblib.load(scalerDir)
    
  #Evaluate

  db = (resolution[2] - resolution[1]) / resolution[0]    # bin width in discriminator distribution
  bins = np.arange(resolution[1], resolution[2]+db, db)   # bin edges in discriminator distribution

  ###########################
  # Read and evaluate signals
  ###########################
  
  statInfoSig = {}
  #Infos about statistic

  Signal = []
  for s in signal:
    x, y = pickBenchmark(s)
    df, weight = loadDataFrame(os.path.join(inputDirSig, s+'/'), preselection, variables, weights, lumi)
    statInfoSig[s]=df.shape[0]
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
  
  statInfoBkg = {}
  #Infos about statistic
  
  totBkgEvents = 0.
  totBkgVar = 0.
  Background = []
  for b in background:
    df, weight = loadDataFrame(os.path.join(inputDirBkg, b+'/'), preselection, variables, weights, lumi)
    statInfoBkg[b]=df.shape[0]
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
    #print s['sig']
    print s['m_stop'], s['m_X'], s['sig'].max(), bins[np.where(s['sig'] == s['sig'].max())]

  x = np.array([s['m_stop'] for s in Signal], dtype=float)
  y = np.array([s['m_X'] for s in Signal], dtype=float)
  z = np.array([s['sig_max'] for s in Signal],dtype=float)

  #print x, y, z
  # Set up a regular grid of interpolation points
  fig, ax1 = plt.subplots(figsize=(8,6))
  xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
  xi, yi = np.meshgrid(xi, yi)

  # Interpolate
  rbf = scipy.interpolate.LinearNDInterpolator(points=np.array((x, y)).T, values=z)
  zi = rbf(xi, yi)

  im = ax1.imshow(zi, vmin=0., vmax=5., origin='lower',
             extent=[x.min(), x.max(), y.min(), y.max()])
  cbar = plt.colorbar(im)
  cbar.set_label('Significance')
  ax1.set_xlabel(r'$m_{\tilde{t}}$')
  ax1.set_xlim([x.min(), x.max()])
  ax1.set_ylabel(r'$m_{\chi}$')
  ax1.set_ylim([y.min(), y.max()])
  plt.scatter(x, y, c='black')
  plt.plot(x, x-84., color='black')
  plt.plot(x, x-175., color='black')
  AtlasStyle_mpl.ATLASLabel(ax1, 0.022, 0.925, 'Work in progress')
  AtlasStyle_mpl.LumiLabel(ax1, 0.022, 0.875, lumi=lumi*0.001)
  #plt.show()
  
  if save:
        if not os.path.exists('./plots/'):
            os.makedirs('./plots/')
            print('Creating folder plots')
        plt.savefig('plots/'+fileName+'_evaluated_grid.pdf')
        plt.savefig('plots/'+fileName+'_evaluated_grid.png')
        plt.close()
  
  diag_165 = {}
  diag_150 = {}
  diag_120 = {}
  diag_90 = {}
  
  for key, value in statInfoSig.iteritems():
      x, y = pickBenchmark(key)
      deltaM = float(x)-float(y)
      if deltaM==165.0:
          diag_165[x]=value
      elif deltaM==150.0:
          diag_150[x]=value
      elif deltaM==120.0:
          diag_120[x]=value
      elif deltaM==90.0:
          diag_90[x]=value
      else:
          print 'Error: Unknown diagonal in evaluate_signalGrid'
          return 0 
  
  sortedLabels165 = sorted(diag_165)
  sortedLabels150 = sorted(diag_150)
  sortedLabels120 = sorted(diag_120)
  sortedLabels90 = sorted(diag_90)
  
  values_165 = []
  values_150 = []
  values_120 = []
  values_90 = []
  
  for label in sortedLabels165:
      values_165.append(diag_165[label])

  for label in sortedLabels150:
      values_150.append(diag_150[label])
      
  for label in sortedLabels120:
      values_120.append(diag_120[label])
      
  for label in sortedLabels90:
      values_90.append(diag_90[label])
      
  csignal = sum(values_90)+sum(values_120)+sum(values_150)+sum(values_165)
  trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
      
  signalP = mpatches.Patch(color='None', label='signal: ' + str(csignal))
  ttbar = mpatches.Patch(color='None', label=r'$t\overline{t}$: ' + str(statInfoBkg['mc16d_ttbar']))
  singletop = mpatches.Patch(color='None', label= 'single top: '+ str(statInfoBkg['mc16d_singletop']))
  Wjets = mpatches.Patch(color='None', label= r'$W$ + jets: '+ str(statInfoBkg['mc16d_Wjets']))
  tps = mpatches.Patch(color='None', label='params(t): ' + str(trainable_count)) #Trainable parameters
  
  #print sortedLabels90, sortedLabels120, sortedLabels150
  #print values_90, values_120, values_150
  
  plt.figure('statistic')
  d165 = plt.plot(sortedLabels165, values_165, 'b-x',label=r'$\Delta M = 165$ GeV')
  d150 = plt.plot(sortedLabels150, values_150, 'b-x',label=r'$\Delta M = 150$ GeV')
  d120 = plt.plot(sortedLabels120, values_120, 'r-x',label=r'$\Delta M = 120$ GeV')
  d90 = plt.plot(sortedLabels90, values_90, 'g-x', label=r'$\Delta M = 90$ GeV')
  plt.xlabel(r'$m_{\tilde{t}}$ [GeV]')
  plt.ylabel('Statistic')
  plt.title('Statistic of samples')
  plt.legend(loc='best', handles=[d165[0],d150[0],d120[0],d90[0],signalP,ttbar,singletop,Wjets,tps])
  
  if save:
        if not os.path.exists('./plots/'):
            os.makedirs('./plots/')
            print('Creating folder plots')
        plt.savefig('plots/'+fileName+'_StatisticTraining.pdf')
        plt.savefig('plots/'+fileName+'_StatisticTraining.png')
        plt.close()
        
        filepath = 'plots/' + fileName + '_StatisticTrainingValues.txt'
        infofile = open(filepath, 'w')
        infofile.write('M165: ' + ';'.join(sortedLabels165) + ' ' +';'.join([str(i) for i in values_165])+'\n')
        infofile.write('M150: ' + ';'.join(sortedLabels150) + ' ' +';'.join([str(i) for i in values_150])+'\n')
        infofile.write('M120: ' + ';'.join(sortedLabels120) + ' ' + ';'.join([str(i) for i in values_120])+'\n')
        infofile.write('M90: ' + ';'.join(sortedLabels90) + ' '+ ';'.join([str(i) for i in values_90]))
        infofile.close()
        
        
########### Evaluate on different cuts than in dataset

def evaluate_signalGridCuts(modelDir, resolution=np.array([50,0,1], dtype=float), save=False, fileName='Test'):
  print('Evaluating singal grid...') 
  
  if fileName=='Grid_test':
      fileName=modelDir.replace('TrainedModels/models/','').replace('.h5','')
    
  infofile = open(modelDir.replace('.h5','_infofile.txt'))
  infos = infofile.readlines()
  
  #Parse Strings for correct datatypes
  
  variables=infos[4].replace('Used variables for training: ','').replace('\n','').split()
  weights=infos[5].replace('Used weights: ', '').replace('\n','').split()
  lumi=float(infos[7].replace('Used Lumi: ','').replace('\n',''))
  background=infos[9].replace('Used background files: ','').replace('; \n','').replace(' ','').split(';')
  
  preselection = preselection_evaluate
  
  print 'Using the following preselection to evaluate:' , preselection
  
  signal = ['stop_bWN_250_100', 'stop_bWN_250_130', 'stop_bWN_250_160', 'stop_bWN_300_150', 'stop_bWN_300_180', 'stop_bWN_300_210', 'stop_bWN_350_185', 'stop_bWN_350_200', 'stop_bWN_350_230', 'stop_bWN_350_260', 'stop_bWN_400_235', 'stop_bWN_400_250', 'stop_bWN_400_280', 'stop_bWN_400_310', 'stop_bWN_450_285', 'stop_bWN_450_300', 'stop_bWN_450_330', 'stop_bWN_450_360', 'stop_bWN_500_335', 'stop_bWN_500_350', 'stop_bWN_500_380', 'stop_bWN_550_385', 'stop_bWN_550_400', 'stop_bWN_550_430', 'stop_bWN_550_460', 'stop_bWN_600_435', 'stop_bWN_600_450', 'stop_bWN_600_480', 'stop_bWN_600_510', 'stop_bWN_650_485', 'stop_bWN_650_500', 'stop_bWN_650_530', 'stop_bWN_650_560']
  
  #Get Scaler and model from modelDir
   
  model = load_model(modelDir)
  
  scalerDir=modelDir.replace('.h5','_scaler.pkl')
  scaler=joblib.load(scalerDir)
    
  #Evaluate

  db = (resolution[2] - resolution[1]) / resolution[0]    # bin width in discriminator distribution
  bins = np.arange(resolution[1], resolution[2]+db, db)   # bin edges in discriminator distribution

  ###########################
  # Read and evaluate signals
  ###########################

  Signal = []
  for s in signal:
    x, y = pickBenchmark(s)
    df, weight = loadDataFrame(os.path.join(inputDirSig, s+'/'), preselection, variables, weights, lumi)
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
  for b in background:
    df, weight = loadDataFrame(os.path.join(inputDirBkg, b+'/'), preselection, variables, weights, lumi)
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
  fig, ax1 = plt.subplots(figsize=(8,6))
  xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
  xi, yi = np.meshgrid(xi, yi)

  # Interpolate
  rbf = scipy.interpolate.LinearNDInterpolator(points=np.array((x, y)).T, values=z)
  zi = rbf(xi, yi)

  im = ax1.imshow(zi, vmin=0., vmax=5., origin='lower',
             extent=[x.min(), x.max(), y.min(), y.max()])
  cbar = plt.colorbar(im)
  cbar.set_label('Significance')
  ax1.set_xlabel(r'$m_{\tilde{t}}$')
  ax1.set_xlim([x.min(), x.max()])
  ax1.set_ylabel(r'$m_{\chi}$')
  ax1.set_ylim([y.min(), y.max()])
  plt.scatter(x, y, c='black')
  plt.plot(x, x-84., color='black')
  plt.plot(x, x-175., color='black')
  AtlasStyle_mpl.ATLASLabel(ax1, 0.022, 0.925, 'Work in progress')
  AtlasStyle_mpl.LumiLabel(ax1, 0.022, 0.875, lumi=lumi*0.001)
  #plt.show()
  
  if save:
        if not os.path.exists('./plots/'):
            os.makedirs('./plots/')
            print('Creating folder plots')
        isFile = True
        n = 1
        while isFile:
            filepath = './plots/' + fileName + '_evaluated_grid_cuts_' + str(n) + '_infofile.txt'
            if os.path.isfile(filepath) and filepath.endswith('.txt'):
                n += 1
                isFile=True
            else: 
                isFile=False
                infofile = open(filepath, 'w')
                print('Saving evaluation informations to ' , filepath)
                presels = ''
                for pre in preselection_evaluate:
                    if pre['type'] == 'condition':
                        presels += pre['name'] + '-threshold: ' + str(pre['threshold']) + ' type: ' + pre['type'] + ' variable: ' + pre['variable'] + ' lessthan: ' + str(pre['lessthan']) + ' and morethan: ' +  str(pre['morethan']) + '; '
                    else:
                        presels += pre['name'] + '-threshold: ' + str(pre['threshold']) + ' type: ' + pre['type'] + '; '
                infofile.write('Used preselection for evaluation: ' + presels)
                infofile.close()            
        plt.savefig('plots/'+fileName+'_evaluated_grid_cuts_' + str(n) + '.pdf')
        plt.savefig('plots/'+fileName+'_evaluated_grid_cuts_' + str(n) + '.png')
        plt.close()
