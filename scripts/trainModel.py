#!/usr/bin/env python

# main script for training
# define samples, variables and algorithms in config/ directory
import os, sys, copy
import timer

import numpy as np

from collections import namedtuple
from LoadData import prepDataset, loadFromRoot
from models import trainBDT, trainNN

Sample = namedtuple('Sample', 'name dataframe')

def checkDataset(y_train, y_test, w_train, w_test, multiclass=False):
  # Print number of weighted and unweighted events
  nb=0; ns=0; nb_w=0; ns_w=0
  if not multiclass:
    for i in range(0, len(y_train)):
      if (y_train[i]<0.5):
        nb += 1.
        nb_w += w_train[i]
      if (y_train[i]>0.5):
        ns += 1.
        ns_w += w_train[i]
    ns_train = ns; nsw_tain = ns_w
    nb_train = nb; nbw_train = nb_w
    print "Number of unweighted training events (sig/bkg): %0.2f / %0.2f"%(ns, nb)
    print "Number of weighted training events (sig/bkg): %0.2f / %0.2f"%(ns_w, nb_w)
    for i in range(0, len(y_test)):
      if (y_test[i]<0.5):
        nb += 1.
        nb_w += w_test[i]
      if (y_test[i]>0.5):
        ns += 1.
        ns_w += w_test[i]

  elif multiclass:
    for i in range(0, len(y_train)):
      if (y_train[i]>0.5):
        nb += 1.
        nb_w += w_train[i]
      if (y_train[i]<0.5):
        ns += 1.
        ns_w += w_train[i]
    ns_train = ns; nsw_tain = ns_w
    nb_train = nb; nbw_train = nb_w
    print "Number of unweighted training events (sig/bkg): %0.2f / %0.2f"%(ns, nb)
    print "Number of weighted training events (sig/bkg): %0.2f / %0.2f"%(ns_w, nb_w)
    for i in range(0, len(y_test)):
      if (y_test[i]>0.5):
        nb += 1.
        nb_w += w_test[i]
      if (y_test[i]<0.5):
        ns += 1.
        ns_w += w_test[i]

  print "Total unweighted events (sig/bkg): %0.2f / %0.2f"%(ns, nb)
  print "Total weighted events (sig/bkg): %0.2f / %0.2f"%(ns_w, nb_w)
  print "Ratio of training-to-testing samples (sig/bkg): %0.2f / %0.2f"%(ns_train/ns, nb_train/nb)

def saveModel(model, modelDir, modelName, alg):
  print 'Saving model and weights...'
  if alg.lower() == 'bdt':
    joblib.dump(model, modelDir+modelName+'.h5')
    joblib.dump(model, modelDir+modelName+'.pkl')
  else:
    model.save(modelDir+modelName+'.h5', overwrite=True)
    model.save_weights(modelDir+modelName+'_weights.h5', overwrite=True)
    # save as JSON
    json_string = model.to_json()
    open(modelDir+modelName+'.json', 'w').write(json_string)
    # save as YAML
    yaml_string = model.to_yaml()
    open(modelDir+modelName+'.yaml', 'w').write(json_string)

def parse_options():
  import argparse

  parser = argparse.ArgumentParser()
  #parser.add_argument('-C', '--config', help="Config file", default="python/loadConfig.py")
  #parser.add_argument('-a', '--analysis', help="name of the analysis to run" , default="NN")
  parser.add_argument('-m', '--multiclass', help="Multi-Classification (True/False)" , default=True, type=bool)
  parser.add_argument('-s', '--saveDataset', help="Save the training data, if it exists, it will load existing file!" , default='./TrainedModels/models/defaultDataset.h5')

  opts = parser.parse_args()

  return opts

def main():
  # define timer to check how long the job runs
  t = timer.Timer()
  t.start()

  opts = parse_options()

  #ConfigFile = opts.config
  data = opts.saveDataset

  print "Loading config file!"
  from loadConfig import Signal, Background, preselection, nvar, lumi, weights, saveDir, fileSuffix, analysis

  #if not os.path.exists(data):
  #  sig = []; bkg = []
  #  print 'Loading signal...'
  #  for s in Signal:
  #    sig.append(Sample(s[0], loadFromRoot(loadDir+s[0]+'/', s[1], preselection, nvar, weights, lumi)))
  #  print 'Loading background...'
  #  for b in Background:
  #    bkg.append(Sample(b[0], loadFromRoot(loadDir+b[0]+'/', b[1], preselection, nvar, weights, lumi)))
  #else:
  #  sig = Signal
  #  bkg = Background

  print "Creating training and test set!"
  X_train, X_test, y_train, y_test, w_train, w_test = prepDataset(Signal, Background, data, multiclass=opts.multiclass)

  checkDataset(y_train, y_test, w_train, w_test, multiclass=opts.multiclass)
  
  for ana in analysis:
    if (ana.name.lower() == 'bdt'): 
      model_BDT, y_pred_BDT = trainBDT(X_train, X_test, y_train, y_test, w_train, w_test, ana.options['classifier'], ana.options['max_depth'], ana.options['min_samples_leaf'], ana.options['n_estimators'], ana.options['learning_rate'])
      saveModel(model_BDT, '', ana.modelname, alg=ana.name)
    if (ana.name.lower() == 'nn'):
      model_DNN, y_pred_DNN = trainNN(X_train, X_test, y_train, y_test, w_train, w_test, ana.options['layers'], ana.options['ncycles'], ana.options['batchSize'], ana.options['dropout'], ana.options['optimizer'], ana.options['activation'], ana.options['initializer'], ana.options['learningRate'], ana.options['decay'], ana.options['momentum'], ana.options['nesterov'], ana.options['multiclassification'])
      saveModel(model_DNN, '', ana.modelname, alg=ana.name)
    #elif (ana.name.lower() == 'rnn'): trainRNN()

  # end timer and print time
  t.stop()
  t0 = t.elapsed
  t.reset()

  hour = t0 // 3600
  t0 %= 3600
  minutes = t0 // 60
  t0 %= 60
  seconds = t0

  print '-----Runtime Summary -----'
  print 'Job ran %d h:%d min:%d sec' % ( hour, minutes, seconds)
  print '--------------------------'  
  
if __name__ == '__main__':
  main()
