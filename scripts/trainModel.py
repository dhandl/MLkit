#!/usr/bin/env python

# main script for training
# define samples, variables and algorithms in config/ directory
import os, sys, copy
from datetime import datetime
import timer

import pandas as pd
import numpy as np

from prepareTraining import prepareTraining
from models import trainBDT, trainNN

from sklearn.externals import joblib

from collections import namedtuple
Sample = namedtuple('Sample', 'name dataframe')

def getModel(models, modeltype):
  for m in models:
    if m.name == modeltype:
      model = m
  try:
    return model
  except NameError:
    sys.exit("Type {} not found in config/algorithm.py!".format(modeltype))

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

def saveModel(model, modelDir, weightDir, modelName, alg):
  print 'Saving model {} and weights...'.format(alg)
  if alg.lower() == 'bdt':
    joblib.dump(model, os.path.join(modelDir,modelName+'.h5'))
    #joblib.dump(model, os.path.join(modelDir,modelName+'.pkl'))
  else:
    model.save(os.path.join(modelDir,modelName+'.h5'), overwrite=True)
    model.save_weights(os.path.join(weightDir,modelName+'_weights.h5'), overwrite=True)
    # save as JSON
    json_string = model.to_json()
    open(os.path.join(modelDir,modelName+'.json'), 'w').write(json_string)
    # save as YAML
    yaml_string = model.to_yaml()
    open(os.path.join(modelDir,modelName+'.yaml'), 'w').write(json_string)

def runtimeSummary(t0):
  hour = t0 // 3600
  t0 %= 3600
  minutes = t0 // 60
  t0 %= 60
  seconds = t0

  print '-----Runtime Summary -----'
  print 'Job ran %d h:%d min:%d sec' % ( hour, minutes, seconds)
  print '--------------------------'  
 
def parse_options():
  import argparse

  workdir = os.getenv('WorkDir')
  output = os.path.join(workdir, 'TrainedModels')

  parser = argparse.ArgumentParser()
  #parser.add_argument('-C', '--config', help="Config file", default="python/loadConfig.py")
  parser.add_argument('-a', '--analysis', help="Name of the algorithm to run" , default="NN")
  parser.add_argument('-d', '--dataset', help="Name of the dataset file")
  parser.add_argument('-m', '--multiclass', help="Multi-Classification (True/False)" , default=False, type=bool)
  parser.add_argument('-n', '--name', help="Name of the output files")
  parser.add_argument('-o', '--output', help="Directory for output files" , default=output)
  parser.add_argument('-r', '--reproduce', help='Constant seed for reproducabilty', default=False, type=bool)
  parser.add_argument('-t', '--trainsize', help='Size of training data. Both (float/int) possible', default=None)
  parser.add_argument('-u', '--testsize', help='Size of test data. Both (float/int) possible', default=None)

  opts = parser.parse_args()

  opts.weightDir = os.path.join(opts.output, 'weights')
  opts.modelDir = os.path.join(opts.output, 'models')
  opts.dataDir = os.path.join(opts.output, 'datasets')
  opts.plotDir = os.path.join(opts.output, 'plots')

  if not os.path.exists(opts.weightDir):
    os.makedirs(opts.weightDir)
  if not os.path.exists(opts.modelDir):
    os.makedirs(opts.modelDir)
  if not os.path.exists(opts.dataDir):
    os.makedirs(opts.dataDir)
  if not os.path.exists(opts.plotDir):
    os.makedirs(opts.plotDir)

  if not opts.name:
    opts.name =  datetime.now().strftime("%Y-%m-%d_%H-%M_")

  if type(opts.trainsize) is str: 
    if '.' in opts.trainsize:
      opts.trainsize = float(opts.trainsize)
    else:
      opts.trainsize = int(opts.trainsize)
  if type(opts.testsize) is str: 
      if '.' in opts.testsize:
        opts.testsize = float(opts.testsize)
      else:
        opts.testsize = int(opts.testsize)
        
  return opts

def main():
  # define timer to check how long the job runs
  t = timer.Timer()
  t.start()

  opts = parse_options()

  print "Loading configuration..."
  from variables import preselection, lumi, nvar, weight
  from samples import Signal, Background
  from algorithm import analysis
  
  alg = getModel(analysis, opts.analysis)
  opts.name = opts.name + alg.modelname
  
  dataset = os.path.join(opts.dataDir,opts.dataset+'.h5')
  
  print "Creating training and test set!"
  X_train, X_test, y_train, y_test, w_train, w_test = prepareTraining(Signal, Background, preselection, nvar, weight, dataset, lumi, opts.trainsize, opts.testsize, opts.reproduce, multiclass=opts.multiclass)

  checkDataset(y_train, y_test, w_train, w_test, multiclass=opts.multiclass)
  
  if (opts.analysis.lower() == 'bdt'): 
      model, y_pred = trainBDT(X_train, X_test, y_train, y_test, w_train, w_test, alg.options['classifier'], alg.options['max_depth'], alg.options['min_samples_leaf'], alg.options['n_estimators'], alg.options['learning_rate'], opts.reproduce)
  elif (opts.analysis.lower() == 'nn'):
      model, history, y_pred = trainNN(X_train, X_test, y_train, y_test, w_train, w_test, alg.options['layers'], alg.options['ncycles'], alg.options['batchSize'], alg.options['dropout'], alg.options['optimizer'], alg.options['activation'], alg.options['initializer'], alg.options['learningRate'], alg.options['decay'], alg.options['momentum'], alg.options['nesterov'], alg.options['multiclassification'])
    from plotting.plot_learning_curve import learning_curve_for_keras
    learning_curve_for_keras(history, opts.plotDir, opts.name)
    #elif (opts.analysis.lower() == 'rnn'):
    #  trainRNN()

  saveModel(model, opts.modelDir, opts.weightDir, opts.name, opts.analysis)

  # end timer and print time
  t.stop()
  t0 = t.elapsed
  t.reset()
  runtimeSummary(t0)
  
if __name__ == '__main__':
  main()
