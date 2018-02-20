import os

import numpy as np
import h5py

def getDataset(filename):
  h5f = h5py.File(filename)
  # full dataset
  X = h5f['X'][:]
  y = h5f['y'][:]
  w = h5f['w'][:]
  ix = h5f['ix'][:]
  split = h5f['train_test_frac'][:]
  # splitted dataset
  X_train = h5f['X_train'][:]
  y_train = h5f['y_train'][:]
  w_train = h5f['w_train'][:]
  ix_train = h5f['ix_train'][:]
  X_test = h5f['X_test'][:]
  y_test = h5f['y_test'][:]
  w_test = h5f['w_test'][:]
  ix_test = h5f['ix_test'][:]
  
  # TODO: add indices ix_...
  collection = {'X':X, 'y':y, 'w':w, 'ix':ix, 'split_frac':split,\
                'X_train':X_train, 'y_train':y_train, 'w':w_train, 'ix':ix_train,\
                'X_test':X_test, 'y_test':y_test, 'w':w_test, 'ix':ix_test}

  #collection = {'X':X, 'y':y, 'w':w, 'ix':ix, 'split_frac':split,\
  #              'X_train':X_train, 'y_train':y_train, 'w_train':w_train,\
  #              'X_test':X_test, 'y_test':y_test, 'w_test':w_test,}
  
  return collection 

#def reshuffleDataset():
# if one wants to apply different splitting...   
