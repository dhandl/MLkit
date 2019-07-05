#!/usr/bin/env python

import os, sys

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py

import AtlasStyle_mpl

from copy import deepcopy

from prepareTraining import prepareTraining
from prepareSequentialTraining import prepareSequentialTraining

import keras.backend as K
from keras.models import load_model

import tensorflow as tf

from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc, roc_auc_score

from getRatio import getRatio


COLLECTION = ['jet'] 
REMOVE_VAR = ['_m', '_mv2c10', '_id', '0_pt', '0_eta', '0_phi', '0_e', '1_pt', '1_eta', '1_phi', '1_e']


def getImportanceBySquaredWeight(model, nvar, rnn=False):

  weights = model.get_layer('dense_1').get_weights()

  if rnn:
    rnn_units = weights[0].shape[0] - len(nvar)  # count the units of the RNN
  else:
    rnn_units = 0

  importance = []
  lstm = rnn_units*['lstm']   # create a list of the size of the RNN units
 
  for i in xrange(len(lstm+nvar)):
    importance.append(( (lstm+nvar)[i] , np.sum(weights[0][i,:]**2) ))

  importance = sorted(importance[rnn_units:], key=lambda x: x[1])
  return importance


def getImportanceByWeight(model, nvar, rnn=False):

  weights = model.get_layer('dense_1').get_weights()

  if rnn:
    rnn_units = weights[0].shape[0] - len(nvar)  # count the units of the RNN
  else:
    rnn_units = 0

  importance = []
  lstm = rnn_units*['lstm']   # create a list of the size of the RNN units
 
  for i in xrange(len(lstm+nvar)):
    importance.append(( (lstm+nvar)[i] , np.sum(abs(weights[0][i,:])) ))

  importance = sorted(importance[rnn_units:], key=lambda x: x[1])
  return importance


def getImportanceByGradient(model, nvar, X_train, collection=None, rnn=False):
  sess = K.get_session()

  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  
  grads = K.gradients(model.output[:,0], model.input)
  
  if rnn:
    grad_vals = sess.run(grads[1], feed_dict={model.input[0]: collection[0], model.input[1]: X_train})
  else:
    grad_vals = sess.run(grads[0], feed_dict={model.input: X_train})

  importance = []
  for i in xrange(len(nvar)):
    #importance.append( (nvar[i], grad_vals[:,i].mean() * X_train[:,i].std() ) )
    importance.append( (nvar[i], np.abs(grad_vals[:,i]).mean() ) )
  
  importance = sorted(importance, key=lambda x: x[1])

  return importance


def main():
  global COLLECTION
  global REMOVE_VAR  

  # Check number of arguments and act respectively thereof
  if len(sys.argv) == 2:
    modelfile = sys.argv[1:][0]
  else:
    print 'Usage: reevaluate_byShuffle.py <model> (omit directory and file suffix)'
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
  nvar = infos[5].replace('Used variables for training: ', '').replace('\n','')
  nvar = nvar.split()

  recurrent = False
  if analysis.lower() == 'rnn':
    recurrent = True

  h5f = h5py.File(dataset+'.h5','r')
  X_train = h5f['X_train'][:]
  y = h5f['y_train'][:]
  
  y_train = deepcopy(y)
  y_train[y != 0] = 0.
  y_train[y == 0] = 1.

  collection = []
  if recurrent:
    for col in COLLECTION:
      collection.append(h5f['X_train_'+col][:])

  h5f.close()  

  #where_nan = np.isnan(X_train)
  #X_train[where_nan] = -999. 
  X_train = scaler.transform(X_train)   # collection already standardized in training

  print '#----MODEL----#'
  print modelDir
  print model.summary()
  
  ######################################
  # Read in trained and tested dataset #
  ######################################

  if recurrent:
    y_hat = model.predict(collection + [X_train])
  else:
    y_hat = model.predict(X_train)
  
  importanceBySquaredWeight = getImportanceBySquaredWeight(model, nvar, recurrent)
  importanceByWeight = getImportanceByWeight(model, nvar, recurrent)
  impotanceByGrad = getImportanceByGradient(model, nvar, X_train, collection, recurrent) 

  print 100*'#'
  print '\n\t\t\tVariable ranking'
  print '\n sum of squared weights \t sum of absolute weights \t gradients '
  print 100*'-'
  for i in xrange(len(nvar)):
    print '{}: {}\t{}: {}\t{}: {}'.format(importanceBySquaredWeight[i][0], importanceBySquaredWeight[i][1], importanceByWeight[i][0], importanceByWeight[i][1], impotanceByGrad[i][0], impotanceByGrad[i][1]) 
  print 100*'-'
  print 100*'#'

  sys.exit()
  # Re-shuffle for re-evaluate
  X_train_reshuffled = []
  for idx, var in enumerate(nvar):
    X = X_train.copy()
    print X[:1]
    X[:,idx] = np.random.permutation(X[:,idx])
    print X[:1],'\n'
    X = scaler.transform(X)
    X_train_reshuffled.append(X.copy())
    del X

  roc = []
  auc = []

  for i in xrange(len(X_train_reshuffled)):
    print type(X_train_reshuffled[i])
    if recurrent:
      y_predict = model.predict(collection + [X_train_reshuffled[i]]) 
    else:
      y_predict = model.predict(X_train_reshuffled[i])
    
    roc.append(roc_curve(y_train, y_predict[:,0]))
    auc.append(roc_auc_score(y_train, y_predict[:,0]))
    del y_predict

  roc.append(roc_curve(y_train, y_hat[:,0]))
  auc.append(roc_auc_score(y_train, y_hat[:,0]))
  print auc,'\n',importanceBySquaredWeight,'\n',importanceByWeight,'\n',impotanceByGrad,'\n' 

  print 100*'#'
  print '\n\t\t\tVariable ranking'
  print '\n sum of squared weights \t sum of absolute weights \t gradients \t AUC (after shuffle)'
  print 100*'-'
  for i in xrange(len(nvar)):
    print '{}: {}\t{}: {}\t{}: {}\t{}: {}'.format(importanceBySquaredWeight[i][0], importanceBySquaredWeight[i][1], importanceByWeight[i][0], importanceByWeight[i][1], impotanceByGrad[i][0], impotanceByGrad[i][1], nvar[i], auc[i])
  print 100*'-'
  print 100*'#'

  print('Plotting the ROC curves ...')
  fig = plt.figure(figsize=(8,6))
  ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=4)
  ax1.set_xlim((0, 1))
  ax1.set_ylim((0, 1))
  ax1.set_xlabel('$\epsilon_{Sig.}$', horizontalalignment='right', x=1.0)
  ax1.set_ylabel("$r_{Bkg.}$", horizontalalignment='right', y=1.0)

  for i in xrange(len(roc)): 
    try:
      plt.plot(roc[i][1], 1- roc[i][0], '-', label='w/o %s (AUC = %0.4f)'%(nvar[i],auc[i]))
    except IndexError:
      plt.plot(roc[i][1], 1- roc[i][0], '-', label='Default (AUC = %0.4f)'%(auc[i]))
    
  plt.plot([0, 1], [1, 0], '--', color=(0.6, 0.6, 0.6), label='Luck')
  leg = plt.legend(loc="lower left", frameon=False)

  AtlasStyle_mpl.ATLASLabel(ax1, 0.13, 0.9, 'Work in progress')
  #AtlasStyle_mpl.LumiLabel(ax1, 0.02, 0.3, lumi=LUMI*0.001)

  plt.savefig("plots/"+modelfile+"_ROC_n-1_v2.pdf")
  plt.savefig("plots/"+modelfile+"_ROC_n-1_v2.png")
  plt.close()

if __name__ == "__main__":
    main()
   

