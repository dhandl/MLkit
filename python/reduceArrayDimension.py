#!usr/bin/env python

import os, sys

import numpy as np
import h5py

def reduceDimension():

  filename = "20181220_stop_bWN_450_300TRUTH_allBkgs_njet4_nbjet1_met230_mt110_RNN"

  h5f = h5py.File('TrainedModels/dataset/'+filename+'.h5', 'r')

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
  

if __name__ == "__main__":
  reduceDimension()

