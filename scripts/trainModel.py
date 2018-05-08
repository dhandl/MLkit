#!/usr/bin/env python

# main script for training
# define samples, variables and algorithms in config/ directory
import os, sys, copy
from datetime import datetime
import timer
import pickle

import pandas as pd
import numpy as np

from prepareTraining import prepareTraining
from prepareSequentialTraining import prepareSequentialTraining
from models import trainBDT, trainNN, trainRNN

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

from startPlot import startPlot

from collections import namedtuple
Sample = namedtuple('Sample', 'name dataframe')

def getModel(models, modeltype):
  for m in models:
    if m.name == modeltype:
      model = m
  try:
    return model
  except NameError:
    sys.exit('Type {} not found in config/algorithm.py!'.format(modeltype))

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
    print 'Number of unweighted training events (sig/bkg): %0.2f / %0.2f'%(ns, nb)
    print 'Number of weighted training events (sig/bkg): %0.2f / %0.2f'%(ns_w, nb_w)
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
    print 'Number of unweighted training events (sig/bkg): %0.2f / %0.2f'%(ns, nb)
    print 'Number of weighted training events (sig/bkg): %0.2f / %0.2f'%(ns_w, nb_w)
    for i in range(0, len(y_test)):
      if (y_test[i]>0.5):
        nb += 1.
        nb_w += w_test[i]
      if (y_test[i]<0.5):
        ns += 1.
        ns_w += w_test[i]

  print 'Total unweighted events (sig/bkg): %0.2f / %0.2f'%(ns, nb)
  print 'Total weighted events (sig/bkg): %0.2f / %0.2f'%(ns_w, nb_w)
  print 'Ratio of training-to-testing samples (sig/bkg): %0.2f / %0.2f'%(ns_train/ns, nb_train/nb)

def saveModel(model, modelDir, weightDir, modelName, alg):
  print 'Saving model {} and weights...'.format(alg)
  if alg.lower() == 'bdt':
    joblib.dump(model, os.path.join(modelDir,modelName+'.h5'))
    #joblib.dump(model, os.path.join(modelDir,modelName+'.pkl'))
  else:
    model.save(os.path.join(modelDir,modelName+'.h5'), overwrite=True)
    model.save_weights(os.path.join(weightDir,modelName+'_weights.h5'), overwrite=True)
    # save as JSON
    #json_string = model.to_json()
    #open(os.path.join(modelDir,modelName+'.json'), 'w').write(json_string)
    # save as YAML
    #yaml_string = model.to_yaml()
    #open(os.path.join(modelDir,modelName+'.yaml'), 'w').write(json_string)
    
def saveInfos(Imodel, Ianalysis, Idataset, Ivariables, Ipreselection, Ilumi, Isignal, Ibackground, Ialgorithmparams, Itrainsize, Itestsize, Ireproduce, Imulticlass, Iweights):
    '''
    Imodel: Name of the model.h5 file
    Ianalysis: Used Method, e.g. BDT, DNN
    Ialgorithmparams: Parametes of used analysis method
    Imulticlass: Multiclassification (boolean)
    Idataset: Name of the dataset.h5 file
    Ivariables: Variables used for training
    Ipreselection: Used cuts in preselection
    Ilumi: Luminosity
    Isignal: Used signal files for dataset
    Ibackground: Used background files for datset
    Itrainsize: Size of trainset (float)
    Itestsize: Size of testset (float)
    Ireproduce: Repdroduce (boolean)
    Iweights: Used weights for the model
    '''
    print('Saving model informations in infofile...')
    filepath = './TrainedModels/models/' + Imodel + '_infofile.txt'
    infofile = open(filepath, 'w')
    infofile.write('Used analysis method: ' + Ianalysis + '\n')
    infofile.write('Used parameters for this analysis algorithm: ' + Ialgorithmparams + '\n')
    infofile.write('Used multiclass: ' + str(Imulticlass) + '\n')
    infofile.write('Used dataset: ' + Idataset + '\n')
    infofile.write('Used variables for training: ' + Ivariables + '\n')
    infofile.write('Used weights: ' + Iweights + '\n')
    presels = ''
    for pre in Ipreselection:
        presels += pre['name'] + '-threshold: ' + str(pre['threshold']) + ' type: ' + pre['type'] + '; '
    infofile.write('Used preselection: ' + presels + '\n')
    infofile.write('Used Lumi: ' + str(Ilumi) + '\n')
    sigs = ''
    for sig in Isignal:
        sigs += sig['name'] + '; '
    infofile.write('Used signal files: ' + sigs + '\n')
    bkgs = ''
    for bkg in Ibackground:
        bkgs += bkg['name'] + '; '
    infofile.write('Used background files: ' + bkgs + '\n')
    infofile.write('Used trainsize/testsize: ' + str(Itrainsize) + '/' + str(Itestsize) + '\n')
    infofile.write('Used reproduce: ' + str(Ireproduce) + '\n')
    infofile.close()

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
  #parser.add_argument('-C', '--config', help='Config file', default='python/loadConfig.py')
  parser.add_argument('-a', '--analysis', help='Name of the algorithm to run' , default='NN')
  parser.add_argument('-d', '--dataset', help='Name of the dataset file')
  parser.add_argument('-m', '--multiclass', help='Multi-Classification (True/False)' , default=False, type=bool)
  parser.add_argument('-n', '--name', help='Name of the output files')
  parser.add_argument('-o', '--output', help='Directory for output files' , default=output)
  parser.add_argument('-r', '--reproduce', help='Constant seed for reproducabilty', default=False, type=bool)
  parser.add_argument('-t', '--trainsize', help='Size of training data. Both (float/int) possible', default=None)
  parser.add_argument('-u', '--testsize', help='Size of test data. Both (float/int) possible', default=None)
  parser.add_argument('-p', '--plot', help='Plotting the output (True/False)', default=False,type=bool)

  opts = parser.parse_args()

  opts.weightDir = os.path.join(opts.output, 'weights')
  opts.modelDir = os.path.join(opts.output, 'models')
  opts.dataDir = os.path.join(opts.output, 'datasets')

  if not os.path.exists(opts.weightDir):
    os.makedirs(opts.weightDir)
  if not os.path.exists(opts.modelDir):
    os.makedirs(opts.modelDir)
  if not os.path.exists(opts.dataDir):
    os.makedirs(opts.dataDir)

  if not opts.name:
    opts.name = datetime.now().strftime('%Y-%m-%d_%H-%M_')

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

  print 'Loading configuration...'
  from variables import preselection, lumi, nvar, weight
  from samples import Signal, Background
  from algorithm import analysis
  
  alg = getModel(analysis, opts.analysis)
  opts.name = opts.name + alg.modelname
  
  dataset = os.path.join(opts.dataDir,opts.dataset+'.h5')
  
  print 'Creating training and test set!'
  if (opts.analysis.lower() == 'rnn'):
    X_train, X_test, y_train, y_test, w_train, w_test, sequence = prepareSequentialTraining(Signal, Background, preselection, alg.options['collection'], nvar, weight, dataset, lumi, opts.trainsize, opts.testsize, opts.reproduce, multiclass=opts.multiclass)
    
  else:
    X_train, X_test, y_train, y_test, w_train, w_test = prepareTraining(Signal, Background, preselection, nvar, weight, dataset, lumi, opts.trainsize, opts.testsize, opts.reproduce, multiclass=opts.multiclass)

  checkDataset(y_train, y_test, w_train, w_test, multiclass=opts.multiclass)
  
  if (opts.analysis.lower() == 'bdt'): 
    model, y_pred = trainBDT(X_train, X_test, y_train, y_test, w_train, w_test, alg.options['classifier'], alg.options['max_depth'],                              alg.options['min_samples_leaf'], alg.options['n_estimators'], alg.options['learning_rate'], 
                             opts.reproduce)

  elif (opts.analysis.lower() == 'nn'):
      
    print 'Standardize training and test set...'
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)  
      
    model, history, y_pred = trainNN(X_train, X_test, y_train, y_test, w_train, w_test, alg.options['layers'], 
                                     alg.options['ncycles'], alg.options['batchSize'], alg.options['dropout'], 
                                     alg.options['optimizer'], alg.options['activation'], alg.options['initializer'], alg.options['regularizer'], alg.options['classWeight'], 
                                     alg.options['learningRate'], alg.options['decay'], alg.options['momentum'], 
                                     alg.options['nesterov'], alg.options['multiclassification'])

    with open(os.path.join(opts.modelDir,opts.name+'_history.pkl'), 'w') as hist_pi:
      pickle.dump(history.history, hist_pi)

  elif (opts.analysis.lower() == 'rnn'):
      
    if alg.options['mergeModels']:
        print 'Standardize training set...'
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    model, history, y_pred = trainRNN(X_train, X_test, y_train, y_test, w_train, w_test, sequence, alg.options['collection'],
                                      alg.options['unit_type'], alg.options['n_units'], alg.options['combinedDim'],
                                      alg.options['epochs'], alg.options['batchSize'], alg.options['dropout'], 
                                      alg.options['optimizer'], alg.options['activation'], alg.options['initializer'], alg.options['regularizer'], 
                                      alg.options['learningRate'], alg.options['decay'], 
                                      alg.options['momentum'], alg.options['nesterov'], alg.options['mergeModels'], 
                                      alg.options['multiclassification'])

    with open(os.path.join(opts.modelDir,opts.name+'_history.pkl'), 'w') as hist_pi:
      pickle.dump(history.history, hist_pi)

  saveModel(model, opts.modelDir, opts.weightDir, opts.name, opts.analysis)
  
  saveInfos(opts.name, opts.analysis.lower(), opts.dataset, ' '.join(nvar), preselection, lumi, Signal, Background, str(alg.options), opts.trainsize, opts.testsize, opts.reproduce, opts.multiclass, ' '.join(weight))
  
  try:
    print('Saving Scaler to file...')
    joblib.dump(scaler, os.path.join(opts.modelDir,opts.name+'_scaler.pkl'))
  except NameError:
      print('No Scaler found')

  # end timer and print time
  t.stop()
  t0 = t.elapsed
  t.reset()
  runtimeSummary(t0)
  
  if opts.plot:
    print('Start Plotting...')
    startPlot(os.path.join('TrainedModels/models',opts.name+'.h5'),save=True)    
  
if __name__ == '__main__':
  main()
