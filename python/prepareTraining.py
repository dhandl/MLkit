import os, copy, sys
import random 

# for arrays
import pandas as pd
import numpy as np
import h5py

# scikit-learn
from sklearn.model_selection import train_test_split

from collections import namedtuple
Sample = namedtuple('Sample', 'name dataframe')


def prepareTraining(sigList, bkgList, preselection, nvar, weight, output, lumi=100e3, trainSize=None, testSize=None, reproduce=False, multiclass=False):
  if os.path.isfile(output) and output.endswith('.h5'):
    print 'Loading existing dataset from: {}'.format(output)
    h5f = h5py.File(output, 'r')
    split = h5f['train_test_frac'][:]
    ix = h5f['ix'][:]
    X = h5f['X'][:]
    y = h5f['y'][:]
    w = h5f['w'][:]
    X_train = h5f['X_train'][:]
    X_test = h5f['X_test'][:]
    y_train = h5f['y_train'][:]
    y_test = h5f['y_test'][:]
    w_train = h5f['w_train'][:]
    w_test = h5f['w_test'][:]
    ix_train = h5f['ix_train'][:]
    ix_test = h5f['ix_test'][:]
    h5f.close() 

  if not os.path.exists(output):
    if reproduce:
      print 'Warning! Constant seed is activated.'
      random_state = 14
      np.random.seed(random_state)
    else: random_state = None;
    if trainSize is None:
      print 'Warning! Size of training set not specified. Set training data to 0.5 of complete dataset.'
      trainSize = 0.5
      if testSize is None:
        testSize = trainSize
    else:
      print 'Fraction of training set: {}'.format(trainSize) if type(trainSize) is float else 'Number training events: {}'.format(trainSize)
      print 'Fraction of test set: {}'.format(testSize) if type(testSize) is float else 'Number test events: {}'.format(testSize)
    
    Signal = []
    Background = []
    for s in sigList:
      print 'Loading signal {} from {}...'.format(s['name'], s['path'])
      Signal.append(Sample(s['name'], loadDataFrame(s['path'], preselection, nvar, weight, lumi)))
      
    for b in bkgList:
      print 'Loading background {} from {}...'.format(b['name'], b['path'])
      Background.append(Sample(b['name'], loadDataFrame(b['path'], preselection, nvar, weight, lumi)))
  
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
        
    for i, b in enumerate(Background):
      i = i + 1
      bkg = np.concatenate((bkg, b.dataframe[0]))
      bkg_w = np.concatenate((bkg_w, b.dataframe[1]))
      bkg_y = np.concatenate((bkg_y, np.full(b.dataframe[0].shape[0], i)))
    
    X = np.concatenate((sig, bkg))
    w = np.concatenate((sig_w, bkg_w))
    if multiclass:
      y = np.concatenate((sig_y, bkg_y))
    else:
      y = []
      for _df, ID in [(sig, 0), (bkg, 1)]:
        y.extend([ID] * _df.shape[0])
      y = np.array(y)
   
    ix = range(X.shape[0])
    X_train, X_test, y_train, y_test, w_train, w_test, ix_train, ix_test = train_test_split(X, y, w, ix, train_size=trainSize, test_size=testSize, random_state=random_state)  

    split = np.array([X_train.shape[0]/X.shape[0], X_test.shape[0]/X.shape[0]])  
    h5f = h5py.File(output, 'w')
    h5f.create_dataset('train_test_frac', data=split)
    h5f.create_dataset('ix', data=ix)
    h5f.create_dataset('X', data=X)
    h5f.create_dataset('y', data=y)
    h5f.create_dataset('w', data=w)
    h5f.create_dataset('X_train', data=X_train)
    h5f.create_dataset('X_test', data=X_test)
    h5f.create_dataset('y_train', data=y_train)
    h5f.create_dataset('y_test', data=y_test)
    h5f.create_dataset('w_train', data=w_train)
    h5f.create_dataset('w_test', data=w_test)
    h5f.create_dataset('ix_train', data=ix_train)
    h5f.create_dataset('ix_test', data=ix_test)
    h5f.close()  

  return (X_train, X_test, y_train, y_test, w_train, w_test)


def loadDataFrame(path, cut=None, nvar=None, weights=[], lumi=100e3):
  if os.path.isdir(path):
    files = os.listdir(path)
  else:
    path, pattern = os.path.split(path)
    fList = os.listdir(path)
    files = []
    for f in fList:
      if f.startswith(pattern):
        files.append(f)

  if len(files) == 1:
    f = files[0]
    print f
    if os.path.isfile(os.path.join(path, f)) and f.endswith(".h5"):
      _df = pd.read_hdf(os.path.join(path, f))
      df = applyCut(_df, cut)
      df = selectVarList(df, nvar+weights)
      del _df
  elif len(files)>1:
    first = True
    for f in files:
      print f
      if os.path.isfile(os.path.join(path, f)) and f.endswith(".h5"):
        if first:
          _df = pd.read_hdf(os.path.join(path, f))
          df = applyCut(_df, cut)
          df = selectVarList(df, nvar+weights)
          del _df
          first = False
        else:
          _df = pd.read_hdf(os.path.join(path, f))
          df_slice = applyCut(_df, cut)
          df_slice = selectVarList(df_slice, nvar+weights)
          df = pd.concat((df, df_slice), ignore_index=True)
          del _df, df_slice

  df, weight_df = np.split(df, [len(nvar)], axis=1)
  weight_df = weightFrame(weight_df, weights, lumi)
  return df, weight_df


def weightFrame(df, weights, lumi=100e3):
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
      elif c['type'] == 'not':
        df = df[ df[c['name']] != c['threshold'] ]
      elif c['type'] == 'condition':
        df = df[(df[c['name']] < c['threshold']) | ((df[c['name']] > c['threshold']) & (df[c['variable']] < c['lessthan']) | (df[c['variable']] > c['morethan']))]
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
