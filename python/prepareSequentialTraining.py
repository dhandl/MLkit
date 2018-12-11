import os, copy, sys
import random 

# for arrays
import pandas as pd
import numpy as np
import h5py

# scikit-learn
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ShuffleSplit

from collections import namedtuple
Sample = namedtuple('Sample', 'name dataframe')


def prepareSequentialTraining(output, sigList=None, bkgList=None, preselection=None, col=None, removeVar=None, nvar=None, weight=None, lumi=100e3, kFold = None, trainSize=None, testSize=None, reproduce=False, multiclass=False):
  if (kFold) and (os.path.isfile(output.replace('.h5','_kFoldCV0.h5'))):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    w_train = []
    w_test = []
    ix_train = []
    ix_test = []
    collection = []

    for i in range(kFold):
      if os.path.isfile(output.replace('.h5','_kFoldCV'+str(i)+'.h5')) and output.endswith('.h5'):
        print 'Loading existing dataset from: {}'.format(output.replace('.h5','_kFoldCV'+str(i)+'.h5'))
        h5f = h5py.File(output.replace('.h5','_kFoldCV'+str(i)+'.h5'), 'r')
        X_train.append(h5f['X_train'][:])
        X_test.append(h5f['X_test'][:])
        y_train.append(h5f['y_train'][:])
        y_test.append(h5f['y_test'][:])
        w_train.append(h5f['w_train'][:])
        w_test.append(h5f['w_test'][:])
        ix_train.append(h5f['ix_train'][:])
        ix_test.append(h5f['ix_test'][:])
        sequence = []
        for idx, c in enumerate(col):
          sequence.append({'name':c, 'X_train':h5f['X_train_'+c][:], 'X_test':h5f['X_test_'+c][:]})
        h5f.close()
        collection.append(sequence) 
  
    return (X_train, X_test, y_train, y_test, w_train, w_test, collection) 
        
     
  else:
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
      sequence = []
      for idx, c in enumerate(col):
        sequence.append({'name':c, 'X_train':h5f['X_train_'+c][:], 'X_test':h5f['X_test_'+c][:]})
      h5f.close() 
      return (X_train, X_test, y_train, y_test, w_train, w_test, sequence)

  if not os.path.exists(output):
    if reproduce:
      print 'Warning! Constant seed is activated.'
      random_state = 14
      np.random.seed(random_state)
    else: 
      random_state = None
    if kFold:
      print '{}-fold cross-validation applied'.format(kFold)
      print 'Given train and test size is omitted!'
    elif (trainSize is None) and (kFold is None):
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
      Signal.append(Sample(s['name'], loadDataFrame(s['path'], preselection, col, removeVar, nvar, weight, lumi)))
      
    for b in bkgList:
      print 'Loading background {} from {}...'.format(b['name'], b['path'])
      Background.append(Sample(b['name'], loadDataFrame(b['path'], preselection, col, removeVar, nvar, weight, lumi)))
  
    sig = np.empty([0, Signal[0].dataframe[0].shape[1]])
    sig_w = np.empty(0)
    sig_y = np.empty(0)
    bkg = np.empty([0, Background[0].dataframe[0].shape[1]])
    bkg_w = np.empty(0)
    bkg_y = np.empty(0)
    
    sequence = []
    for idx, c in enumerate(col):
      sequence.append({'name':c, 'vars':Signal[0].dataframe[2][idx]['vars'], 'df_full':pd.DataFrame()}) 

    for s in Signal:
      sig = np.concatenate((sig, s.dataframe[0]))
      sig_w = np.concatenate((sig_w, s.dataframe[1]))
      sig_y = np.concatenate((sig_y, np.zeros(s.dataframe[0].shape[0])))
      for idx, seq in enumerate(sequence):
        seq['df_full'] = pd.concat((seq['df_full'], s.dataframe[2][idx]['df']), ignore_index=True)

    for i, b in enumerate(Background):
      i = i + 1
      bkg = np.concatenate((bkg, b.dataframe[0]))
      bkg_w = np.concatenate((bkg_w, b.dataframe[1]))
      bkg_y = np.concatenate((bkg_y, np.full(b.dataframe[0].shape[0], i)))
      for idx, seq in enumerate(sequence):
        seq['df_full'] = pd.concat((seq['df_full'], b.dataframe[2][idx]['df']), ignore_index=True)
    
    X = np.concatenate((sig, bkg))
    w = np.concatenate((sig_w, bkg_w))
    if multiclass:
      y = np.concatenate((sig_y, bkg_y))
    else:
      y = []
      for _df, ID in [(sig, 0), (bkg, 1)]:
        y.extend([ID] * _df.shape[0])
      y = np.array(y)
   
    if kFold:
      X_train = []
      X_test = []
      y_train = []
      y_test = []
      w_train = []
      w_test = []
      ix_train = []
      ix_test = []
      collection = []

      sss = StratifiedShuffleSplit(n_splits=kFold, test_size=(1./kFold), random_state=random_state)
      cv = 0
      ix = range(X.shape[0])

      for train_index, test_index in sss.split(X, y):
        X_tr, X_te = X[train_index], X[test_index]
        y_tr, y_te = y[train_index], y[test_index]
        w_tr, w_te = w[train_index], w[test_index]

        X_train.append(X_tr)
        X_test.append(X_te)
        y_train.append(y_tr)
        y_test.append(y_te)
        w_train.append(w_tr)
        w_test.append(w_te)
        ix_train.append(train_index)
        ix_test.append(test_index)

        split = np.array([X_tr.shape[0]/X.shape[0], X_te.shape[0]/X.shape[0]])
        outfile = output.replace(".h5", "_kFoldCV"+str(cv)+".h5")
        h5f = h5py.File(outfile, 'w')
        h5f.create_dataset('train_test_frac', data=split)
        h5f.create_dataset('ix', data=ix)
        h5f.create_dataset('X', data=X.astype(float))
        h5f.create_dataset('y', data=y.astype(float))
        h5f.create_dataset('w', data=w.astype(float))
        h5f.create_dataset('X_train', data=X_tr.astype(float))
        h5f.create_dataset('X_test', data=X_te.astype(float))
        h5f.create_dataset('y_train', data=y_tr.astype(float))
        h5f.create_dataset('y_test', data=y_te.astype(float))
        h5f.create_dataset('w_train', data=w_tr.astype(float))
        h5f.create_dataset('w_test', data=w_te.astype(float))
        h5f.create_dataset('ix_train', data=train_index)
        h5f.create_dataset('ix_test', data=test_index)
        
        for seq in sequence:
          seq['n_max'] = max([len(j) for j in seq['df_full'][seq['name']+'_pt']])
          seq['X_train'], seq['X_test'] = create_stream(seq['df_full'], train_index, test_index, seq['n_max'], sort_col=seq['name']+'_pt', VAR_FILE_NAME=outfile) 
        collection.append(sequence)

        for seq in sequence:
          h5f.create_dataset('X_train_'+seq['name'], data=seq['X_train'])
          h5f.create_dataset('X_test_'+seq['name'], data=seq['X_test'])
        h5f.close()       
        cv = cv+1
  
      return (X_train, X_test, y_train, y_test, w_train, w_test, collection) 

    else:
      ix = range(X.shape[0])
      X_train, X_test, y_train, y_test, w_train, w_test, ix_train, ix_test = train_test_split(X, y, w, ix, train_size=trainSize, test_size=testSize, random_state=random_state)  
  
      split = np.array([X_train.shape[0]/X.shape[0], X_test.shape[0]/X.shape[0]])
      h5f = h5py.File(output, 'w')
      h5f.create_dataset('train_test_frac', data=split)
      h5f.create_dataset('ix', data=ix)
      h5f.create_dataset('X', data=X.astype(float))
      h5f.create_dataset('y', data=y.astype(float))
      h5f.create_dataset('w', data=w.astype(float))
      h5f.create_dataset('X_train', data=X_train.astype(float))
      h5f.create_dataset('X_test', data=X_test.astype(float))
      h5f.create_dataset('y_train', data=y_train.astype(float))
      h5f.create_dataset('y_test', data=y_test.astype(float))
      h5f.create_dataset('w_train', data=w_train.astype(float))
      h5f.create_dataset('w_test', data=w_test.astype(float))
      h5f.create_dataset('ix_train', data=ix_train)
      h5f.create_dataset('ix_test', data=ix_test)
  
      for seq in sequence:
        seq['n_max'] = max([len(j) for j in seq['df_full'][seq['name']+'_pt']])
        #seq['X_train'], seq['X_test'] = create_stream(seq['df_full'], range(X_train.shape[0]), range(X_test.shape[0]), seq['n_max'], sort_col=seq['name']+'_pt', VAR_FILE_NAME=output) 
        seq['X_train'], seq['X_test'] = create_stream(seq['df_full'], ix_train, ix_test, seq['n_max'], sort_col=seq['name']+'_pt', VAR_FILE_NAME=output) 
    
      for seq in sequence:
        h5f.create_dataset('X_train_'+seq['name'], data=seq['X_train'])
        h5f.create_dataset('X_test_'+seq['name'], data=seq['X_test'])
      h5f.close()  
      return (X_train, X_test, y_train, y_test, w_train, w_test, sequence)


def loadDataFrame(path, cut=None, col=None, rm=None, nvar=None, weights=[], lumi=100e3):
  files = os.listdir(path)
  collection = []
  for c in col:
    collection.append({'name':c})
  if len(files) == 1:
    f = files[0]
    print f
    if os.path.isfile(path+f) and f.endswith(".h5"):
      _df = pd.read_hdf(path+f)
      sequence = []
      for c in collection:
        c['vars'] = [key for key in _df.keys() if key.startswith(c['name'])]
        for r in rm:
          try:
            c['vars'].remove(c['name']+r)
          except ValueError:
            continue
        sequence += c['vars']
      df = applyCut(_df, cut)
      df = selectVarList(df, nvar+weights+sequence)
      del _df
  elif len(files)>1:
    first = True
    for f in files:
      print f
      if os.path.isfile(path+f) and f.endswith(".h5"):
        if first:
          _df = pd.read_hdf(path+f)
          sequence = []
          for c in collection:
            c['vars'] = [key for key in _df.keys() if key.startswith(c['name'])]
            for r in rm:
              try:
                c['vars'].remove(c['name']+r)
              except ValueError:
                continue
            sequence += c['vars']
          df = applyCut(_df, cut)
          df = selectVarList(df, nvar+weights+sequence)
          del _df
          first = False
        else:
          _df = pd.read_hdf(path+f)
          sequence = []
          for c in collection:
            c['vars'] = [key for key in _df.keys() if key.startswith(c['name'])]
            for r in rm:
              try:
                c['vars'].remove(c['name']+r)
              except ValueError:
                continue
            sequence += c['vars']
          df_slice = applyCut(_df, cut)
          df_slice = selectVarList(df_slice, nvar+weights+sequence)
          df = pd.concat((df, df_slice), ignore_index=True)
          del _df, df_slice

  df, weight_df, sequence_df = np.split(df, [len(nvar),len(nvar)+len(weights)], axis=1)
  weight_df = weightFrame(weight_df, weights, lumi)

  seq_split = []
  for c in collection:
    if len(seq_split) == 0: last = 0;
    else: last = seq_split[-1];
    seq_split.append(last+len(c['vars']))    
  
  sequence_df = np.split(sequence_df, seq_split, axis=1)

  for i,c in enumerate(collection):
    c['df'] = sequence_df[i]

  del sequence_df
  return df, weight_df, collection


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

def create_stream(df, ix_train, ix_test, num_obj, sort_col, VAR_FILE_NAME):
  n_variables = df.shape[1]
  var_names = df.keys()
  data = np.zeros((df.shape[0], num_obj, n_variables), dtype='float32')
  
  # call functions to build X (a.k.a. data)
  sort_objects(df, data, sort_col, num_obj)
  # ix_{train, test} from above or from previously stored ordering
  Xobj_train = data[ix_train]
  Xobj_test = data[ix_test]
  
  # print 'Scaling features ...' 
  scale(Xobj_train, var_names, savevars=True, VAR_FILE_NAME=VAR_FILE_NAME.replace('.h5','_scaling.json')) # scale training sample and save scaling
  scale(Xobj_test, var_names, savevars=False, VAR_FILE_NAME=VAR_FILE_NAME.replace('.h5','_scaling.json')) # apply scaling to test set
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
  import tqdm
  # i = event number, event = all the variables for that event 
  for i, event in tqdm.tqdm(df.iterrows(), total=df.shape[0]): 
    # objs = [[pt's], [eta's], ...] of particles for each event

    #objs = np.array([v.tolist() for v in event.get_values()], dtype='float32')[:, (np.argsort(event[SORT_COL]))[::-1]]
    objs = np.array([v for v in event.get_values()], dtype='float32')[:, (np.argsort(event[SORT_COL]))[::-1]]
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
  import json
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

