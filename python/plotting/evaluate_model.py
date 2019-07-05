#!/usr/bin/env python

import os, sys

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import AtlasStyle_mpl

from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc, roc_auc_score

from prepareTraining import loadDataFrame, weightFrame, selectVarList, applyCut, varHasIndex, pickIndex
from prepareSequentialTraining import loadDataFrame as loadSequentialDataFrame
from getRatio import getRatio

from collections import namedtuple
Sample = namedtuple('Sample', 'name' 'path')

sys.path.append('./python/plotting/')
import plot_ROCcurves

inputDir = '/project/etp5/dhandl/samples/SUSY/Stop1L/FullRun2/hdf5/cut_mt30_met60_preselection_v60/'

DIR = 'TrainedModels/models/'

SAVEDIR = '/project/etp5/dhandl/plots/Stop1L/FullRun2/ML/'
if not os.path.exists(SAVEDIR):
  os.makedirs(SAVEDIR)

FILENAME = 'concurrentMethods_bWN_650_500'

logScale = False
if logScale:
  FILENAME = FILENAME + '_log'

BWN_PRESEL_BKG = 10673.12
#BWN_PRESEL_SIG = 397.53
BWN_PRESEL_SIG = 169.34

# 500_380
#WP = [
#      {'name':'LHCP17',                  'legend':True, 'bkg':257.04,'sig':54.83, 'color':'lightgrey'},
#      {'name':'LHCP17 optimised',        'legend':True, 'bkg':77.66, 'sig':27.70, 'color':'grey'},
#      {'name':'cut-based',               'legend':True, 'bkg':22.45, 'sig':31.31, 'color':'black'},
#      {'name':'alt. NN',                 'legend':False, 'bkg':20.89, 'sig':31.87, 'color':'blue'},
#      {'name':'NN (incl. 4 lead. jets)', 'legend':False, 'bkg':6.04,  'sig':13.88, 'color':'dodgerblue'},
#      {'name':'BDT',                     'legend':False, 'bkg':5.22,  'sig':13.69, 'color':'green'},
#      {'name':'NN (w/o jet RNN)',        'legend':False, 'bkg':5.57,  'sig':14.88, 'color':'orange'},
#      {'name':'RNN',                     'legend':False, 'bkg':4.56,  'sig':16.21, 'color':'red'},
#]

# 650_500
WP = [
      {'name':'LHCP17',                  'legend':True, 'bkg':257.04,'sig':19.12, 'color':'lightgrey'},
      {'name':'LHCP17 optimised',        'legend':True, 'bkg':77.66, 'sig':11.24, 'color':'grey'},
      {'name':'cut-based',               'legend':True, 'bkg':22.45, 'sig':7.74, 'color':'black'},
      {'name':'alt. NN',                 'legend':False, 'bkg':20.89, 'sig':6.99, 'color':'blue'},
      {'name':'NN (incl. 4 lead. jets)', 'legend':False, 'bkg':6.04,  'sig':3.66, 'color':'dodgerblue'},
      {'name':'BDT',                     'legend':False, 'bkg':5.22,  'sig':2.86, 'color':'green'},
      {'name':'NN (w/o jet RNN)',        'legend':False, 'bkg':5.57,  'sig':4.29, 'color':'orange'},
      {'name':'RNN',                     'legend':False, 'bkg':4.56,  'sig':4.96, 'color':'red'},
]

MODELS = [
  {'name':'alt. NN',                  'mdir':'2019-01-07_14-59_DNN_ADAM_leakyReLU_layer128_batch32_ZerosInitializer_noBatchNormalization_l1-0p01_lr0p001_decay0', 'color':'blue'},
  {'name':'NN (incl. 4 lead. jets)',  'mdir':'2019-03-05_18-30_DNN_ADAM_leakyReLU_layer128_batch32_GlorotNormalInitializer_l1-0p01_lr0p0001_decay0p001', 'color':'dodgerblue'},
  {'name':'BDT',                      'mdir':'2019-02-26_17-55_XGBoost_d3_nTrees1000_lr0p1', 'color':'green'},
  {'name':'NN (w/o jet RNN)',         'mdir':'2019-01-30_18-19_DNN_ADAM_leakyReLU_layer128_batch32_GlorotNormalInitializer_l1-0p01_lr0p0001_decay0p001', 'color':'orange'},
  {'name':'RNN',                      'mdir':'2018-12-21_11-20_RNN_jetOnly_ADAM_leakyReLU_LSTM32_128NNlayer_batch32_BatchNorm_NormalInitializer_l2-0p01', 'color':'red'}
]

SIGNAL = [['mc16a_bWN_650_500', 'mc16d_bWN_650_500', 'mc16e_bWN_650_500']]

BACKGROUND = [['mc16a_ttbar', 'mc16d_ttbar', 'mc16e_ttbar'], ['mc16a_singletop', 'mc16d_singletop', 'mc16e_singletop'], ['mc16a_wjets', 'mc16d_wjets', 'mc16e_wjets'], ['mc16a_ttV', 'mc16d_ttV', 'mc16e_ttV'], ['mc16a_multiboson', 'mc16d_multiboson', 'mc16e_multiboson']]

PRESELECTION = [
                {'name':'n_jet',  'threshold':4,      'type':'geq'},
                {'name':'n_bjet',  'threshold':1,      'type':'geq'},
                {'name':'met',    'threshold':230e3,  'type':'geq'},
                {'name':'mt',    'threshold':110e3,  'type':'geq'},
                {'name':'n_lep',  'threshold':1,      'type':'exact'},
                {'name':'lep_pt',  'threshold':25e3,      'type':'geq'}
               ]

WEIGHTS = [
           'weight',
           'lumi_weight',
           'xs_weight',
           'sf_total'
          ]

LUMI = 140e3

RESOLUTION = np.array([50,0,1], dtype=float)
db = (RESOLUTION[2] - RESOLUTION[1]) / RESOLUTION[0]    # bin width in discriminator distribution
bins = np.arange(RESOLUTION[1], RESOLUTION[2]+db, db)   # bin edges in discriminator distribution
center = (bins[:-1] + bins[1:]) / 2

COLLECTION = ['jet'] 
REMOVE_VAR = ['_m', '_mv2c10', '_id', '0_pt', '0_eta', '0_phi', '0_e', '1_pt', '1_eta', '1_phi', '1_e']


def evaluate(model, dataset, scaler, seq_scaler=None, col=None, method='nn'):

  #where_nan = np.isnan(dataset)
  #dataset[where_nan] = -999. 
  if scaler:
    dataset = scaler.transform(dataset)

  if method.lower() == 'rnn':  
    for idx, c in enumerate(col):
      #c['n_max'] = int(max([len(j) for j in c['df'][c['name']+'_pt']]))
      c['n_max'] = 16
      c['Xobj'] = create_scale_stream(c['df'], c['n_max'], sort_col=c['name']+'_pt', VAR_FILE_NAME=seq_scaler) 

    y_hat = model.predict([c['Xobj'] for c in col]+[dataset])

  elif method.lower() == 'nn':
    y_hat = model.predict(dataset)

  elif method.lower() == 'bdt':
    y_hat = model.predict_proba(dataset)

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
 
  for m in MODELS:
    
    modelDir = DIR + m['mdir'] + '.h5'
    DatasetDir = 'TrainedModels/datasets/'

    if os.path.exists(os.path.join(DIR,m['mdir']+'_scaler.pkl')):
      m['scaler'] = joblib.load(os.path.join(DIR,m['mdir']+'_scaler.pkl'))
    else:
      m['scaler'] = None
  
    infofile = open(modelDir.replace('.h5','_infofile.txt'))
    infos = infofile.readlines()
    m['analysis']=infos[0].replace('Used analysis method: ','').replace('\n','')
    m['dataset'] = DatasetDir + infos[3].replace('Used dataset: ', '').replace('\n','')
    m['VAR'] = infos[5].replace('Used variables for training: ', '').replace('\n','').split()

    m['recurrent'] = False
    if m['analysis'].lower() == 'rnn':
      m['recurrent'] = True
      m['seq_scaler'] = m['dataset']+'_scaling.json'
  
    if 'nn' in m['analysis'].lower():
      m['model'] = load_model(os.path.join(DIR, m['mdir']+'.h5'))
    elif 'bdt' in m['analysis'].lower():
      m['model'] = joblib.load(os.path.join(DIR, m['mdir']+'.h5'))

    print '#----MODEL----#'
    print '\t',m['mdir']


    ###########################
    # Read and evaluate signals
    ###########################

    m['Signal'] = []
    for smp in SIGNAL:
      first = True
      for s in smp:
        print 'Sample:\t',s
        x, y = pickBenchmark(s)
        if not m['recurrent']:
          _df, _weight = loadDataFrame(os.path.join(inputDir, s+'/'), PRESELECTION, m['VAR'], WEIGHTS, LUMI)
          print _df.shape,_weight.shape
          if first:
            df = _df.copy()
            weight = _weight.copy()
            first = False
          else: 
            df = pd.concat((df, _df), ignore_index=True)
            weight = pd.concat((weight, _weight), ignore_index=True)
        else: 
          _df, _weight, collection = loadSequentialDataFrame(os.path.join(inputDir, s+'/'), PRESELECTION, COLLECTION, REMOVE_VAR, m['VAR'], WEIGHTS, LUMI)
          print _df.shape,_weight.shape, collection[0]['df'].shape
          if first:
            df = _df.copy()
            weight = _weight.copy()
            seq = collection[0]['df'].copy()
            first = False
          else:
            df = pd.concat((df, _df), ignore_index=True)
            weight = pd.concat((weight, _weight), ignore_index=True)
            seq = pd.concat((seq, collection[0]['df']), ignore_index=True)

      if not m['recurrent']:      
        m['y_pred_sig'] = evaluate(m['model'], df.values, m['scaler'], method=m['analysis'])
        m['y_sig'] = np.ones(m['y_pred_sig'].shape[0])
      else:
        collection[0]['df'] = seq.copy()
        m['y_pred_sig'] = evaluate(m['model'], df.values, m['scaler'], m['seq_scaler'], method=m['analysis'], col=collection)
        m['y_sig'] = np.ones(m['y_pred_sig'].shape[0])
          
      bin_index = np.digitize(m['y_pred_sig'][:,0], bins[1:])   # get the bin index of the output score for each event 
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
      
      m['Signal'].append({'name':s[6:], 'm_stop':x, 'm_X':y, 'dataset':df, 'weight':weight, 'nEvents':weight.sum(), 'outputScore':np.array(outputWeighted), 'outputMC':np.array(outputMC), 'output_var':np.array(outputWeightedVar), 'outputMC_var':np.array(outputMCVar)})

      del df, weight, bin_index, outputWeighted, outputWeightedVar, outputMC, outputMCVar

    ###############################
    # Read and evaluate backgrounds 
    ###############################
   
    m['totBkgEvents'] = 0.
    m['totBkgVar'] = 0.
    m['Background'] = [] 
    for smp in BACKGROUND:
      first = True
      for b in smp:
        print 'Sample:\t',b
        if not m['recurrent']:
          _df, _weight = loadDataFrame(os.path.join(inputDir, b+'/'), PRESELECTION, m['VAR'], WEIGHTS, LUMI)
          print _df.shape,_weight.shape
          if first:
            df = _df.copy()
            weight = _weight.copy()
            first = False
          else:        
            df = pd.concat((df, _df), ignore_index=True)
            weight = pd.concat((weight, _weight), ignore_index=True)
        else:
          _df, _weight, collection = loadSequentialDataFrame(os.path.join(inputDir, b+'/'), PRESELECTION, COLLECTION, REMOVE_VAR, m['VAR'], WEIGHTS, LUMI)
          print _df.shape,_weight.shape, collection[0]['df'].shape
          if first:
            df = _df.copy()
            weight = _weight.copy()
            seq = collection[0]['df'].copy()
            first = False
          else:
            df = pd.concat((df, _df), ignore_index=True)
            weight = pd.concat((weight, _weight), ignore_index=True)
            seq = pd.concat((seq, collection[0]['df']), ignore_index=True)
  
      if not m['recurrent']:
        print df.shape, weight.shape
        m['_'.join(['y_pred',b])] = evaluate(m['model'], df.values, m['scaler'], method=m['analysis'])
        m['_'.join(['y',b])] = np.zeros(m['_'.join(['y_pred',b])].shape[0])
      else:
        collection[0]['df'] = seq
        print df.shape, weight.shape, collection[0]['df'].shape
        m['_'.join(['y_pred',b])] = evaluate(m['model'], df.values, m['scaler'], m['seq_scaler'], method=m['analysis'], col=collection)
        m['_'.join(['y',b])] = np.zeros(m['_'.join(['y_pred',b])].shape[0])

      bin_index = np.digitize(m['_'.join(['y_pred',b])][:,0], bins[1:])
      outputWeighted = []
      outputWeightedVar = []
      outputMC = []
      outputMCVar = []

      m['totBkgEvents'] += weight.sum()
      m['totBkgVar'] += np.sum(weight.values**2.)
      for i in range(len(bins[1:])):
        w = weight.values[np.where(bin_index==i)[0]]
        sigma = np.sum(w**2.)
        outputWeighted.append(w.sum())
        outputWeightedVar.append(sigma)
        outputMC.append(len(w))
        outputMCVar.append(len(w))

      m['Background'].append({'name':b, 'dataset':df, 'weight':weight, 'nEvents':weight.sum(), 'outputScore':np.array(outputWeighted), 'outputMC':np.array(outputMC), 'output_var':np.array(outputWeightedVar), 'outputMC_var':np.array(outputMCVar)})

      del df, weight, bin_index, outputWeighted, outputWeightedVar, outputMC, outputMCVar 
 
    m['totalBkgOutput'] = np.array([b['outputScore'] for b in m['Background']]) 
    m['totalBkgOutput'] = m['totalBkgOutput'].sum(axis=0)
    
    m['totalBkgVar'] = np.array([b['output_var'] for b in m['Background']])
    m['totalBkgVar'] = m['totalBkgVar'].sum(axis=0)
  
    for s in m['Signal']:
      m['roc'] = []
      m['roc_err'] = []

      m['tot_rel'] = np.sqrt(np.sum(s['output_var'])) / s['nEvents']
      for i in range(len(bins[1:])):
        eff_sig = s['outputScore'][i:].sum() / s['nEvents']
        eff_bkg = m['totalBkgOutput'][i:].sum() / m['totalBkgOutput'].sum()
 
        err_sig = np.sqrt(np.sum(s['output_var'][i:])) / s['nEvents']
        err_bkg = np.sqrt(np.sum(m['totalBkgVar'][i:])) / m['totalBkgOutput'].sum()

        if m['totalBkgOutput'][i:].sum() > 0.:
          rel_err_bkg = np.sqrt(np.sum(m['totalBkgVar'][i:])) / m['totalBkgOutput'][i:].sum()
        else:
          rel_err_bkg = 0.
        if s['outputScore'][i:].sum() > 0.:
          rel_err_sig = np.sqrt(np.sum(s['output_var'][i:])) / s['outputScore'][i:].sum()
        else:
          rel_err_sig = 0.
        
        m['total_rel_err'] = np.sqrt(rel_err_bkg**2. + 0.25**2.)

        m['roc'].append((eff_sig, 1-eff_bkg))

        roc_plus_sig = eff_sig + err_sig
        roc_mins_sig = eff_sig - err_sig
        roc_plus_bkg = 1-(eff_bkg + err_bkg)
        roc_mins_bkg = 1-(eff_bkg - err_bkg)

        #roc_err_sig = abs(roc_plus_sig - roc_mins_sig) / 2.
        roc_err_bkg = abs(roc_plus_bkg - roc_mins_bkg) / 2.
        m['roc_err'].append(roc_err_bkg)

      m['roc'] = np.array(m['roc'])
      m['roc_err'] = np.array(m['roc_err'])

    #m['y_bkg'] = np.empty(0)
    #m['y_pred_bkg'] = np.empty(0)

    #for b in BACKGROUND:
    #  m['y_bkg'] = np.concatenate((m['y_bkg'], m['_'.join(['y',b])])) 
    #  m['y_pred_bkg'] = np.concatenate((m['y_pred_bkg'], m['_'.join(['y_pred',b])][:,0]))

    #m['y'] = np.concatenate((m['y_sig'], m['y_bkg']))
    #m['y_pred'] = np.concatenate((m['y_pred_sig'][:,0], m['y_pred_bkg']))

    #m['fpr'], m['tpr'], m['threshold'] = roc_curve(m['y'], m['y_pred'])
    #m['auc'] = roc_auc_score(m['y'], m['y_pred']) 

  print('Plotting ROC curve ...')
  fig = plt.figure(figsize=(8,6))
  ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=4)
  #ax1.set_xlim((bins[0], bins[-1]))
  #ax1.set_ylim((0, 1))
  ax1.set_xlabel('$\epsilon_{Sig.}$', horizontalalignment='right', x=1.0)
  ax1.set_ylabel('$r_{Bkg.}$', horizontalalignment='right', y=1.0)

  for m in MODELS:
    m['auc'] = np.trapz(m['roc'][:,0], m['roc'][:,1], dx=db)
    print 'Area under ROC:\t',m['auc']
    if logScale:
      ax1.set_yscale('log')
      plt.plot(m['roc'][:,0], 1./(1.-m['roc'][:,1]), 'k-', color=m['color'], label='%s (AUC = %0.4f)'%(m['name'], m['auc']))
      plt.fill_between(m['roc'][:,0], 1./(1.-(m['roc'][:,1]-m['roc_err'])), 1./(1.-(m['roc'][:,1]+m['roc_err'])), alpha=0.2, edgecolor=m['color'], facecolor=m['color'], linewidth=0)
      #plt.plot(m['tpr'], 1./m['fpr'], lw=2, label=m['name']+' (AUC = %0.3f)'%(m['auc'])) 
    else:
      plt.plot(m['roc'][:,0], m['roc'][:,1], 'k-', color=m['color'], label='%s (AUC = %0.2f)'%(m['name'], m['auc']))
      plt.fill_between(m['roc'][:,0], (m['roc'][:,1]-m['roc_err']), (m['roc'][:,1]+m['roc_err']), alpha=0.2, edgecolor=m['color'], facecolor=m['color'], linewidth=0)
      #plt.plot(m['tpr'], 1.-m['fpr'], lw=2, label=m['name']+' (AUC = %0.3f)'%(m['auc']))
      ax1.set_xlim((0, 0.16))
      ax1.set_ylim((0.975, 1.0))

  #plt.plot([0, 1], [1, 0], '--', color=(0.6, 0.6, 0.6), label='Luck')


  for p in WP:
    p['eff_sig'] = p['sig']/BWN_PRESEL_SIG
    p['eff_bkg'] = p['bkg']/BWN_PRESEL_BKG
    if p['legend']:
      plt.plot([p['eff_sig']],[1-p['eff_bkg']], '.', color=p['color'], label=p['name'])
    else:
      plt.plot([p['eff_sig']],[1-p['eff_bkg']], '.', color=p['color'])

  leg = plt.legend(loc="lower left", frameon=False)

  #AtlasStyle_mpl.ATLASLabel(ax1, 0.02, 0.25, 'Work in progress')
  AtlasStyle_mpl.Text(ax1, 0.14, 0.52, 'Simulation')
  AtlasStyle_mpl.LumiLabel(ax1, 0.14, 0.46, lumi=LUMI*0.001)

  plt.savefig(SAVEDIR+FILENAME+'.pdf')
  plt.savefig(SAVEDIR+FILENAME+'.png')
  plt.close()

if __name__ == "__main__":
    main()
