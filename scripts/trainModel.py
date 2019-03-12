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
from models import trainBDT, trainNN, trainRNN, trainOptNN

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler

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
        ns += 1.
        ns_w += w_train[i]
      if (y_train[i]>0.5):
        nb += 1.
        nb_w += w_train[i]
    ns_train = ns; nsw_tain = ns_w
    nb_train = nb; nbw_train = nb_w
    print 'Number of unweighted training events (sig/bkg): %0.2f / %0.2f'%(ns, nb)
    print 'Number of weighted training events (sig/bkg): %0.2f / %0.2f'%(ns_w, nb_w)
    for i in range(0, len(y_test)):
      if (y_test[i]<0.5):
        ns += 1.
        ns_w += w_test[i]
      if (y_test[i]>0.5):
        nb += 1.
        nb_w += w_test[i]

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
    infofile.write('Used scaler: ' + Imodel + '_scaler.pkl' + '\n')
    infofile.write('Used variables for training: ' + Ivariables + '\n')
    infofile.write('Used weights: ' + Iweights + '\n')
    presels = ''
    for pre in Ipreselection:
        if pre['type'] == 'condition':
            presels += pre['name'] + '-threshold: ' + str(pre['threshold']) + ' type: ' + pre['type'] + ' variable: ' + pre['variable'] + ' lessthan: ' + str(pre['lessthan']) + ' and morethan: ' +  str(pre['morethan']) + '; '
        else:
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
  parser.add_argument('-k', '--kfold', help='Use k-fold cross-validation', default=None)
  parser.add_argument('-t', '--trainsize', help='Size of training data. Both (float/int) possible', default=None)
  parser.add_argument('-u', '--testsize', help='Size of test data. Both (float/int) possible', default=None)
  parser.add_argument('-p', '--plot', help='Plotting the output (True/False)', default=False,type=bool)
  parser.add_argument('-x', '--hyperoptimization', help='Optimize certain hyperparameters (True/False)', default=False,type=bool)

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

  if type(opts.kfold) is str:
    opts.kfold = int(opts.kfold)
  elif (type(opts.trainsize) is str) and not opts.kfold: 
    if '.' in opts.trainsize:
      opts.trainsize = float(opts.trainsize)
    else:
      opts.trainsize = int(opts.trainsize)
  elif (type(opts.testsize) is str) and not opts.kfold: 
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
    X_train, X_test, y_train, y_test, w_train, w_test, sequence = prepareSequentialTraining(dataset, Signal, Background, preselection, alg.options['collection'], alg.options['removeVar'], nvar, weight, lumi, opts.kfold, opts.trainsize, opts.testsize, opts.reproduce, multiclass=opts.multiclass)
    
  else:
    X_train, X_test, y_train, y_test, w_train, w_test = prepareTraining(dataset, Signal, Background, preselection, nvar, weight, lumi, opts.trainsize, opts.testsize, opts.reproduce, multiclass=opts.multiclass)

  if opts.kfold:
    for i in range(opts.kfold):
      print 'Summary of kfold cross-validation datasets!'
      checkDataset(y_train[i], y_test[i], w_train[i], w_test[i], multiclass=opts.multiclass)
  else:
    checkDataset(y_train, y_test, w_train, w_test, multiclass=opts.multiclass)
  
  if (opts.analysis.lower() == 'bdt'): 
    model, y_pred = trainBDT(X_train, X_test, y_train, y_test, w_train, w_test, alg.options['classifier'], alg.options['max_depth'], alg.options['n_estimators'], alg.options['learning_rate'],  alg.options['lambda'], alg.options['alpha'], alg.options['gamma'], alg.options['scale_pos_weights'], opts.reproduce)

  elif (opts.analysis.lower() == 'nn'):
      
    print 'Standardize training and test set...'
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if opts.hyperoptimization:
        print 'Using hyperas for hyperparameter optimization'
        
        model, y_pred = trainOptNN(X_train, X_test, y_train, y_test, w_train, w_test, alg.options['layers'], 
                                        alg.options['ncycles'], alg.options['batchSize'], alg.options['dropout'], 
                                        alg.options['optimizer'], alg.options['activation'], alg.options['initializer'], alg.options['regularizer'], alg.options['classWeight'], 
                                        alg.options['learningRate'], alg.options['decay'], alg.options['momentum'], 
                                        alg.options['nesterov'], alg.options['multiclassification'], reproduce=opts.reproduce)
        #with open(os.path.join(opts.modelDir,opts.name+'_history.pkl'), 'w') as hist_pi:
            #pickle.dump(history.history, hist_pi)
        
    else:
        model, history, y_pred = trainNN(X_train, X_test, y_train, y_test, w_train, w_test, alg.options['layers'], 
                                        alg.options['ncycles'], alg.options['batchSize'], alg.options['dropout'], 
                                        alg.options['optimizer'], alg.options['activation'], alg.options['initializer'], alg.options['regularizer'], alg.options['classWeight'], 
                                        alg.options['learningRate'], alg.options['decay'], alg.options['momentum'], 
                                        alg.options['nesterov'], alg.options['multiclassification'], reproduce=opts.reproduce)
        
        with open(os.path.join(opts.modelDir,opts.name+'_history.pkl'), 'w') as hist_pi:
            pickle.dump(history.history, hist_pi)

  elif (opts.analysis.lower() == 'rnn'):
      
    if opts.kfold:
      model = []
      history = []
      y_pred = []
      scaler = []
      score = []
     
      for i in range(opts.kfold):
        print 'k-fold cross-validation! Iteration:{}'.format(i)

        if alg.options['mergeModels']:
          print 'Standardize training set...'
          scale = StandardScaler()
          _X_train = scale.fit_transform(X_train[i])
          _X_test = scale.transform(X_test[i])
        
        scaler.append(scale) 

        m, h, y_hat = trainRNN(_X_train, _X_test, y_train[i], y_test[i], w_train[i], w_test[i], sequence[i], alg.options['collection'],
                                        alg.options['unit_type'], alg.options['n_units'], alg.options['combinedDim'],
                                        alg.options['epochs'], alg.options['batchSize'], alg.options['dropout'], 
                                        alg.options['optimizer'], alg.options['activation'], alg.options['initializer'], alg.options['regularizer'], 
                                        alg.options['learningRate'], alg.options['decay'], 
                                        alg.options['momentum'], alg.options['nesterov'], alg.options['mergeModels'], 
                                        alg.options['multiclassification'], alg.options['classWeight'])
        model.append(m)
        history.append(h)
        y_pred.append(y_hat)
        score.append(m.evaluate([seq['X_test'] for seq in sequence[i]]+[X_test[i]], y_test[i]))

        with open(os.path.join(opts.modelDir,opts.name+'_kFoldCV'+str(i)+'_history.pkl'), 'w') as hist_pi:
          pickle.dump(h.history, hist_pi)

    else:
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
                                      alg.options['multiclassification'], alg.options['classWeight'])

      with open(os.path.join(opts.modelDir,opts.name+'_history.pkl'), 'w') as hist_pi:
        pickle.dump(history.history, hist_pi)

  if opts.kfold:
    s = np.array(score, dtype=float)
    print 'Evaluating k-fold cross validation!'
    print 'Score: ', s
    print("\nMean %s: %.2f +/- %.2f" % (model[i].metrics_names[0], np.mean(s[:,0]), np.std(s[:,0])))
    print("\nMean %s: %.2f +/- %.2f" % (model[i].metrics_names[1], np.mean(s[:,1]), np.std(s[:,1])))
    for i in range(opts.kfold):
      saveModel(model[i], opts.modelDir, opts.weightDir, opts.name+'_kFoldCV'+str(i), opts.analysis)
    
      try:
        print('Saving Scaler to file...')
        joblib.dump(scaler[i], os.path.join(opts.modelDir,opts.name+'_kFoldCV'+str(i)+'_scaler.pkl'))
      except NameError:
          print('No Scaler found')
     
      saveInfos(opts.name+'_kFoldCV'+str(i), opts.analysis.lower(), opts.dataset+'_kFoldCV'+str(i), ' '.join(nvar), preselection, lumi, Signal, Background, str(alg.options), opts.trainsize, opts.testsize, opts.reproduce, opts.multiclass, ' '.join(weight))
      
      if opts.plot:
        print('Start Plotting...')
        startPlot(os.path.join('TrainedModels/models',opts.name+'_kFoldCV'+str(i)+'.h5'),save=True, multiclass=opts.multiclass)
    

  else:
    saveModel(model, opts.modelDir, opts.weightDir, opts.name, opts.analysis)
  
    try:
      print('Saving Scaler to file...')
      joblib.dump(scaler, os.path.join(opts.modelDir,opts.name+'_scaler.pkl'))
    except NameError:
        print('No Scaler found')
   
    saveInfos(opts.name, opts.analysis.lower(), opts.dataset, ' '.join(nvar), preselection, lumi, Signal, Background, str(alg.options), opts.trainsize, opts.testsize, opts.reproduce, opts.multiclass, ' '.join(weight))
    
    if opts.plot:
      print('Start Plotting...')
      startPlot(os.path.join('TrainedModels/models',opts.name+'.h5'),save=True, multiclass=opts.multiclass)
    
  # end timer and print time
  t.stop()
  t0 = t.elapsed
  t.reset()
  runtimeSummary(t0)
      

if __name__ == '__main__':
  main()
