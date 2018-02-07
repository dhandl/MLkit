import os, copy, sys
import glob
import random 

import ROOT

from collections import namedtuple

# for arrays
import pandas as pd
import numpy as np
from root_numpy import rec2array, tree2array
import h5py

# scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import json
import tqdm

Sample = namedtuple('Sample', 'name dataframe')

def prepDataset(sigList, bkgList, saveDataset, multiclass=False, fixedTrainSize=-1, fixedTestSize=-1):
  # for Debugging: load an existing training and test sample
  if os.path.isfile(saveDataset) and saveDataset.endswith('.h5'):
    print "Loading previous data set from", saveDataset
    h5f = h5py.File(saveDataset, 'r')
    #ix = h5f['ix'][:]
    #X = h5f['X'][:]
    #y = h5f['y'][:]
    #w = h5f['w'][:]
    X_train = h5f['X_train'][:]
    X_test = h5f['X_test'][:]
    y_train = h5f['y_train'][:]
    y_test = h5f['y_test'][:]
    w_train = h5f['w_train'][:]
    w_test = h5f['w_test'][:]
    h5f.close()

  # Load new samples, create new, independent training and test sample
  if not os.path.exists(saveDataset):
    Signal = []
    Background = []
    for s in sigList:
      print 'Loading signal %s...'%(s['name'])
      Signal.append(Sample(s['name'], loadFromRoot(s['path']+s['name'], s['tree'], s['cut'], s['branches'], s['weights'], s['lumi'])))
    for b in bkgList:
      print 'Loading background %s...'%(b['name'])
      Background.append(Sample(b['name'], loadFromRoot(b['path']+b['name'], b['tree'], b['cut'], b['branches'], b['weights'], b['lumi'])))

    if (type(fixedTrainSize) is int) and (fixedTrainSize > 0):
      print 'Set fixed number of training events! Will randomly pick %i training events for signal and background.'%(fixedTrainSize)
      n_sig = len(Signal)
      n_bkg = len(Background)
      #sig_train_frac = fixedTrainSize/n_sig
      bkg_train_frac = fixedTrainSize/n_bkg
      if fixedTestSize < 0: 
        fixedTestSize = fixedTrainSize
      #sig_test_frac = fixedTestSize/n_sig
      bkg_test_frac = fixedTestSize/n_bkg
      sig = np.empty([0, Signal[0].dataframe[0].shape[1]])
      sig_w = np.empty(0)
      sig_y = np.empty(0)
      sig_train = np.empty([0, Signal[0].dataframe[0].shape[1]])
      bkg_train = np.empty([0, Background[0].dataframe[0].shape[1]])
      sig_train_w = np.empty(0)
      sig_train_y = np.empty(0)
      bkg_train_w = np.empty(0)
      bkg_train_y = np.empty(0)
      sig_test = np.empty([0, Signal[0].dataframe[0].shape[1]])
      bkg_test = np.empty([0, Background[0].dataframe[0].shape[1]])
      sig_test_w = np.empty(0)
      sig_test_y = np.empty(0)
      bkg_test_w = np.empty(0)
      bkg_test_y = np.empty(0)
    else:      
      sig = np.empty([0, Signal[0].dataframe[0].shape[1]])
      sig_w = np.empty(0)
      sig_y = np.empty(0)
      bkg = np.empty([0, Background[0].dataframe[0].shape[1]])
      bkg_w = np.empty(0)
      bkg_y = np.empty(0)

    for s in Signal:
      sig = np.concatenate((sig, s.dataframe[0]))
      sig_w = np.concatenate((sig_w, s.dataframe[1]))
      sig_y = np.concatenate((sig_y, np.zeros(s.dataframe[0].shape[0])))
    if (type(fixedTrainSize) is int) and (fixedTrainSize > 0):
      ix = range(sig.shape[0])
      ix_shuffle = random.sample(ix, len(ix))
      ix_shuffle = np.split(ix_shuffle, [fixedTrainSize, fixedTrainSize+fixedTestSize])
      sig_train = np.concatenate((sig_train, sig[ix_shuffle[0]]))
      sig_train_w = np.concatenate((sig_train_w, sig_w[ix_shuffle[0]]))
      sig_train_y = np.concatenate((sig_train_y, np.zeros(fixedTrainSize)))
      sig_test = np.concatenate((sig_test, sig[ix_shuffle[1]]))
      sig_test_w = np.concatenate((sig_test_w, sig_w[ix_shuffle[1]]))
      sig_test_y = np.concatenate((sig_test_y, np.zeros(fixedTestSize)))
     #if (type(fixedTrainSize) is int) and (fixedTrainSize > 0):
     #   # TODO: Picking the same frac in each signal can cause problems when high m_stop samples have less stats. Weight Factor!
     #   # Currently merging signal together and pick
     #   
     #   ix = range(s.dataframe[0].shape[0])
     #   ix_shuffle = random.sample(ix, len(ix))
     #   ix_shuffle = np.split(ix_shuffle, [sig_train_frac,sig_train_frac+sig_test_frac])
     #   sig_train = np.concatenate((sig_train, s.dataframe[0][ix_shuffle[0]]))
     #   sig_train_w = np.concatenate((sig_train_w, s.dataframe[1][ix_shuffle[0]]))
     #   sig_train_y = np.concatenate((sig_train_y, np.zeros(sig_train_frac)))
     #   sig_test = np.concatenate((sig_test, s.dataframe[0][ix_shuffle[1]]))
     #   sig_test_w = np.concatenate((sig_test_w, s.dataframe[1][ix_shuffle[1]]))
     #   sig_test_y = np.concatenate((sig_test_y, np.zeros(sig_test_frac)))
     #   print sig_test.shape, sig_train.shape
     #else:      
     #  sig = np.concatenate((sig, s.dataframe[0]))
     #  sig_w = np.concatenate((sig_w, s.dataframe[1]))
     #  sig_y = np.concatenate((sig_y, np.zeros(s.dataframe[0].shape[0])))
  
    for i, b in enumerate(Background):
      i = i + 1
      if (type(fixedTrainSize) is int) and (fixedTrainSize > 0):
        ix = range(b.dataframe[0].shape[0])
        ix_shuffle = random.sample(ix, len(ix))
        ix_shuffle = np.split(ix_shuffle, [bkg_train_frac,bkg_train_frac+bkg_test_frac])
        bkg_train = np.concatenate((bkg_train, b.dataframe[0][ix_shuffle[0]]))
        bkg_train_w = np.concatenate((bkg_train_w, b.dataframe[1][ix_shuffle[0]]))
        bkg_train_y = np.concatenate((bkg_train_y, np.full(bkg_train_frac, i)))
        bkg_test = np.concatenate((bkg_test, b.dataframe[0][ix_shuffle[1]]))
        bkg_test_w = np.concatenate((bkg_test_w, b.dataframe[1][ix_shuffle[1]]))
        bkg_test_y = np.concatenate((bkg_test_y, np.full(bkg_test_frac, i)))
      else:
        bkg = np.concatenate((bkg, b.dataframe[0]))
        bkg_w = np.concatenate((bkg_w, b.dataframe[1]))
        bkg_y = np.concatenate((bkg_y, np.full(b.dataframe[0].shape[0], i)))
  
    if (type(fixedTrainSize) is int) and (fixedTrainSize > 0):
      ix = range(2*fixedTrainSize)
      ix_train_shuffle = random.sample(ix, len(ix))
      ix_test_shuffle = random.sample(ix, len(ix))
      X_train = np.concatenate((sig_train, bkg_train))
      X_test = np.concatenate((sig_test, bkg_test))
      w_train = np.concatenate((sig_train_w, bkg_train_w))
      w_test = np.concatenate((sig_test_w, bkg_test_w))
      if multiclass:
        y_train = np.concatenate((sig_train_y, bkg_train_y))
        y_test = np.concatenate((sig_test_y, bkg_test_y))
      else:
        y_train = []; y_test = []
        for _df, ID in [(sig_train, 1), (bkg_train, 0)]:
            y_train.extend([ID] * _df.shape[0])
        for _df, ID in [(sig_test, 1), (bkg_test, 0)]:
            y_test.extend([ID] * _df.shape[0])
        y_train = np.array(y_train); y_test = np.array(y_test)
      X_train = X_train[ix_train_shuffle]
      y_train = y_train[ix_train_shuffle]
      w_train = w_train[ix_train_shuffle]
      X_test = X_test[ix_test_shuffle]
      y_test = y_test[ix_test_shuffle]
      w_test = w_test[ix_test_shuffle]
  
    else:
      X = np.concatenate((sig, bkg))
      w = np.concatenate((sig_w, bkg_w))
      if multiclass:
        y = np.concatenate((sig_y, bkg_y))
      else:
        y = []
        for _df, ID in [(sig, 1), (bkg, 0)]:
          y.extend([ID] * _df.shape[0])
        y = np.array(y)
  
      ix = range(X.shape[0])
      X_train, X_test, y_train, y_test, w_train, w_test, ix_train, ix_test = train_test_split(X, y, w, ix, train_size=0.33, test_size=0.33)
    
    h5f = h5py.File(saveDataset, 'w')
    #h5f.create_dataset('ix', data=ix)
    #h5f.create_dataset('X', data=X)
    #h5f.create_dataset('y', data=y)
    #h5f.create_dataset('w', data=w)
    h5f.create_dataset('X_train', data=X_train)
    h5f.create_dataset('X_test', data=X_test)
    h5f.create_dataset('y_train', data=y_train)
    h5f.create_dataset('y_test', data=y_test)
    h5f.create_dataset('w_train', data=w_train)
    h5f.create_dataset('w_test', data=w_test)
    h5f.close()

  return (X_train, X_test, y_train, y_test, w_train, w_test)

def loadFromRoot(filepath, treename, cut, branches, weights, lumi=1000.):
  files = os.listdir(filepath)
  print 'Loading file(s):'

  if len(files) == 1:
    f = files[0]
    print f
    if os.path.isfile('/'.join([filepath, f])) and f.endswith(".root"):

      infile = ROOT.TFile('/'.join([filepath, f]))
      tree = infile.Get(treename)
      
      source = rec2array(tree2array(tree, branches, cut))
      weight = rec2array(tree2array(tree, weights, cut))
      infile.Close() 
      weight = np.multiply.reduce(weight, axis=1)*lumi 

  elif len(files) > 1:
    source = np.empty([0,len(branches)])
    weight = np.empty([0,len(weights)])

    for f in files:
      print f
      if os.path.isfile('/'.join([filepath, f])) and f.endswith(".root"):    
        
        infile = ROOT.TFile('/'.join([filepath, f]))
        tree = infile.Get(treename)
        temp = rec2array(tree2array(tree, branches, cut))
        tempWeight = rec2array(tree2array(tree, weights, cut))
        source = np.concatenate((source, temp))
        weight = np.concatenate((weight, tempWeight))
        infile.Close()
    
    weight = np.multiply.reduce(weight, axis=1)*lumi

  return (source, weight)

def prepareTraining(sigList, bkgList, preselection, nvar, weight, lumi=100e3, splitFrac=0.33, multiclass=False, fixedTrainSize=-1, fixedTestSize=-1):
  Signal = []
  Background = []
  for s in sigList:
    print 'Loading signal {} from {}...'.format(s['name'], s['path'])
    Signal.append(Sample(s['name'], loadDataFrame(s['path'], preselection, nvar, weight, lumi)))
    
  for b in bkgList:
    print 'Loading background {} from {}...'.format(b['name'], b['path'])
    Background.append(Sample(b['name'], loadDataFrame(b['path'], preselection, nvar, weight, lumi)))

  if (type(fixedTrainSize) is int) and (fixedTrainSize > 0):
    print 'Set fixed number of training events! Will randomly pick %i training events for signal and background.'%(fixedTrainSize)
    n_sig = len(Signal)
    n_bkg = len(Background)
    #sig_train_frac = fixedTrainSize/n_sig
    bkg_train_frac = fixedTrainSize/n_bkg
    if fixedTestSize < 0: 
      fixedTestSize = fixedTrainSize
    #sig_test_frac = fixedTestSize/n_sig
    bkg_test_frac = fixedTestSize/n_bkg
    sig = np.empty([0, Signal[0].dataframe[0].shape[1]])
    sig_w = np.empty(0)
    sig_y = np.empty(0)
    sig_train = np.empty([0, Signal[0].dataframe[0].shape[1]])
    bkg_train = np.empty([0, Background[0].dataframe[0].shape[1]])
    sig_train_w = np.empty(0)
    sig_train_y = np.empty(0)
    bkg_train_w = np.empty(0)
    bkg_train_y = np.empty(0)
    sig_test = np.empty([0, Signal[0].dataframe[0].shape[1]])
    bkg_test = np.empty([0, Background[0].dataframe[0].shape[1]])
    sig_test_w = np.empty(0)
    sig_test_y = np.empty(0)
    bkg_test_w = np.empty(0)
    bkg_test_y = np.empty(0)
  else:      
    sig = np.empty([0, Signal[0].dataframe[0].shape[1]])
    sig_w = np.empty(0)
    sig_y = np.empty(0)
    bkg = np.empty([0, Background[0].dataframe[0].shape[1]])
    bkg_w = np.empty(0)
    bkg_y = np.empty(0)

  for s in Signal:
    sig = np.concatenate((sig, s.dataframe[0]))
    sig_w = np.concatenate((sig_w, s.dataframe[1]))
    sig_y = np.concatenate((sig_y, np.zeros(s.dataframe[0].shape[0])))
  if (type(fixedTrainSize) is int) and (fixedTrainSize > 0):
    ix = range(sig.shape[0])
    ix_shuffle = random.sample(ix, len(ix))
    ix_shuffle = np.split(ix_shuffle, [fixedTrainSize, fixedTrainSize+fixedTestSize])
    sig_train = np.concatenate((sig_train, sig[ix_shuffle[0]]))
    sig_train_w = np.concatenate((sig_train_w, sig_w[ix_shuffle[0]]))
    sig_train_y = np.concatenate((sig_train_y, np.zeros(fixedTrainSize)))
    sig_test = np.concatenate((sig_test, sig[ix_shuffle[1]]))
    sig_test_w = np.concatenate((sig_test_w, sig_w[ix_shuffle[1]]))
    sig_test_y = np.concatenate((sig_test_y, np.zeros(fixedTestSize)))
   #if (type(fixedTrainSize) is int) and (fixedTrainSize > 0):
   #   # TODO: Picking the same frac in each signal can cause problems when high m_stop samples have less stats. Weight Factor!
   #   # Currently merging signal together and pick
   #   
   #   ix = range(s.dataframe[0].shape[0])
   #   ix_shuffle = random.sample(ix, len(ix))
   #   ix_shuffle = np.split(ix_shuffle, [sig_train_frac,sig_train_frac+sig_test_frac])
   #   sig_train = np.concatenate((sig_train, s.dataframe[0][ix_shuffle[0]]))
   #   sig_train_w = np.concatenate((sig_train_w, s.dataframe[1][ix_shuffle[0]]))
   #   sig_train_y = np.concatenate((sig_train_y, np.zeros(sig_train_frac)))
   #   sig_test = np.concatenate((sig_test, s.dataframe[0][ix_shuffle[1]]))
   #   sig_test_w = np.concatenate((sig_test_w, s.dataframe[1][ix_shuffle[1]]))
   #   sig_test_y = np.concatenate((sig_test_y, np.zeros(sig_test_frac)))
   #   print sig_test.shape, sig_train.shape
   #else:      
   #  sig = np.concatenate((sig, s.dataframe[0]))
   #  sig_w = np.concatenate((sig_w, s.dataframe[1]))
   #  sig_y = np.concatenate((sig_y, np.zeros(s.dataframe[0].shape[0])))
  
  for i, b in enumerate(Background):
    i = i + 1
    if (type(fixedTrainSize) is int) and (fixedTrainSize > 0):
      ix = range(b.dataframe[0].shape[0])
      ix_shuffle = random.sample(ix, len(ix))
      ix_shuffle = np.split(ix_shuffle, [bkg_train_frac,bkg_train_frac+bkg_test_frac])
      bkg_train = np.concatenate((bkg_train, b.dataframe[0][ix_shuffle[0]]))
      bkg_train_w = np.concatenate((bkg_train_w, b.dataframe[1][ix_shuffle[0]]))
      bkg_train_y = np.concatenate((bkg_train_y, np.full(bkg_train_frac, i)))
      bkg_test = np.concatenate((bkg_test, b.dataframe[0][ix_shuffle[1]]))
      bkg_test_w = np.concatenate((bkg_test_w, b.dataframe[1][ix_shuffle[1]]))
      bkg_test_y = np.concatenate((bkg_test_y, np.full(bkg_test_frac, i)))
    else:
      bkg = np.concatenate((bkg, b.dataframe[0]))
      bkg_w = np.concatenate((bkg_w, b.dataframe[1]))
      bkg_y = np.concatenate((bkg_y, np.full(b.dataframe[0].shape[0], i)))
  
  if (type(fixedTrainSize) is int) and (fixedTrainSize > 0):
    ix = range(2*fixedTrainSize)
    ix_train_shuffle = random.sample(ix, len(ix))
    ix_test_shuffle = random.sample(ix, len(ix))
    X_train = np.concatenate((sig_train, bkg_train))
    X_test = np.concatenate((sig_test, bkg_test))
    w_train = np.concatenate((sig_train_w, bkg_train_w))
    w_test = np.concatenate((sig_test_w, bkg_test_w))
    if multiclass:
      y_train = np.concatenate((sig_train_y, bkg_train_y))
      y_test = np.concatenate((sig_test_y, bkg_test_y))
    else:
      y_train = []; y_test = []
      for _df, ID in [(sig_train, 1), (bkg_train, 0)]:
          y_train.extend([ID] * _df.shape[0])
      for _df, ID in [(sig_test, 1), (bkg_test, 0)]:
          y_test.extend([ID] * _df.shape[0])
      y_train = np.array(y_train); y_test = np.array(y_test)
    X_train = X_train[ix_train_shuffle]
    y_train = y_train[ix_train_shuffle]
    w_train = w_train[ix_train_shuffle]
    X_test = X_test[ix_test_shuffle]
    y_test = y_test[ix_test_shuffle]
    w_test = w_test[ix_test_shuffle]
  
  else:
    X = np.concatenate((sig, bkg))
    w = np.concatenate((sig_w, bkg_w))
    if multiclass:
      y = np.concatenate((sig_y, bkg_y))
    else:
      y = []
      for _df, ID in [(sig, 1), (bkg, 0)]:
        y.extend([ID] * _df.shape[0])
      y = np.array(y)
  
    ix = range(X.shape[0])
    X_train, X_test, y_train, y_test, w_train, w_test, ix_train, ix_test = train_test_split(X, y, w, ix, train_size=splitFrac, test_size=splitFrac, random_state=14)
  
  return (X_train, X_test, y_train, y_test, w_train, w_test)

def loadDataFrame(path, cut=None, nvar=None, weights=[], lumi=100e3):
  files = os.listdir(path)
  if len(files) == 1:
    f = files[0]
    if os.path.isfile(path+f) and f.endswith(".h5"):
      _df = pd.read_hdf(path+f)
      df = selectVarList(_df, nvar+weights)
      df = applyCut(df, cut)
      del _df
  elif len(files)>1:
    first = True
    for f in files:
      if os.path.isfile(path+f) and f.endswith(".h5"):
        if first:
          _df = pd.read_hdf(path+f)
          df = selectVarList(_df, nvar+weights)
          df = applyCut(df, cut)
          del _df
          first = False
        else:
          _df = pd.read_hdf(path+f)
          df_slice = selectVarList(_df, nvar+weights)
          df_slice = applyCut(df_slice, cut)
          df = pd.concat((df, df_slice), ignore_index=True)
          del _df, df_slice

  df, weight_df = np.split(df, [len(nvar)], axis=1)
  weight_df = weightFrame(weight_df, weights, lumi)
  return df, weight_df

def createFullDataset(fileList, varList):
  df = pd.concat((f[varList] for f in fileList), ignore_index=True)
  return df

def weightFrame(df, weights, lumi=36100.):
  first = True
  for w in weights:
    if first:
      weight = df[w]
      first = False
    else:
      weight = weight * df[w]
  weight = weight * lumi
  return weight

def selectVarList(df, nvar=None):
  if type(nvar)==list:
    first = True
    for v in nvar:
      if varHasIndex(v): 
        index, name = pickIndex(v)
        _df = df[name].str[int(index)]
      else:  
        _df = df[v]
      if first:
        first = False
        new_df = _df
      else:
        new_df = pd.concat((new_df, _df), axis=1)
    new_df.columns = nvar
  del _df
  return new_df

def applyCut(df, cut=None):
  if type(cut) == list:
    for i, c in enumerate(cut):
      if c['type'] == 'exact':
        df = df[ df[c['name']] == c['threshold'] ]
      elif c['type'] == 'less':
        df = df[ df[c['name']] < c['threshold'] ]
      elif c['type'] == 'leq':
        df = df[ df[c['name']] <= c['threshold'] ]
      elif c['type'] == 'greater':
        df = df[ df[c['name']] > c['threshold'] ]
      elif c['type'] == 'geq':
        df = df[ df[c['name']] >= c['threshold'] ]
  return df

def varHasIndex(var):
  if ('[' in var) and (']' in var):
    return True 
  else:
    return False

def pickIndex(var):
  if ('[' in var) and (']' in var):
    name = var.split('[')
    start = var.index('[')
    stop = var.index(']')
    index = ''
    for i in range(start+1, stop):
      index = index + var[i]
  return index, name[0]
 
def create_stream(df, ix_train, ix_test, num_obj, sort_col):
  n_variables = df.shape[1]
  var_names = df.keys()
  data = np.zeros((df.shape[0], num_obj, n_variables), dtype='float32')

  # call functions to build X (a.k.a. data)
  sort_objects(df, data, sort_col, num_obj)
  # ix_{train, test} from above or from previously stored ordering
  Xobj_train = data[ix_train]
  Xobj_test = data[ix_test]
  
  # print 'Scaling features ...'
  scale(Xobj_train, var_names, savevars=True) # scale training sample and save scaling
  scale(Xobj_test, var_names, savevars=False) # apply scaling to test set
  return Xobj_train, Xobj_test

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
  # i = event number, event = all the variables for that event 
  for i, event in tqdm.tqdm(df.iterrows(), total=df.shape[0]): 
    # objs = [[pt's], [eta's], ...] of particles for each event 
    objs = np.array([v.tolist() for v in event.get_values()], dtype='float32')[:, (np.argsort(event[SORT_COL]))[::-1]]
    # total number of tracks per jet      
    nobjs = objs.shape[1] 
    # take all tracks unless there are more than n_tracks 
    data[i, :(min(nobjs, max_nobj)), :] = objs.T[:(min(nobjs, max_nobj)), :] 
    # default value for missing tracks 
    data[i, (min(nobjs, max_nobj)):, :  ] = -999

def scale(data, var_names, savevars, VAR_FILE_NAME='scaling.json'):
  ''' 
  Args:
  -----
      data: a numpy array of shape (nb_events, nb_particles, n_variables)
      var_names: list of keys to be used for the model
      savevars: bool -- True for training, False for testing
                it decides whether we want to fit on data to find mean and std 
                or if we want to use those stored in the json file 
  
  Returns:
  --------
      modifies data in place, writes out scaling dictionary
  '''
  scale = {}
  if savevars: 
    for v, name in enumerate(var_names):
      #print 'Scaling feature %s of %s (%s).' % (v + 1, len(var_names), name)
      f = data[:, :, v]
      slc = f[f != -999]
      m, s = slc.mean(), slc.std()
      slc -= m
      slc /= s
      data[:, :, v][f != -999] = slc.astype('float32')
      scale[name] = {'mean' : float(m), 'sd' : float(s)}
      
    with open(VAR_FILE_NAME, 'wb') as varfile:
      json.dump(scale, varfile)
  else:
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

