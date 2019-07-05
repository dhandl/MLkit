#!/usr/bin/env python

import os, sys

import pandas as pd
import numpy as np

import ROOT
ROOT.gSystem.Load('libRooStats')

from prepareTraining import loadDataFrame, weightFrame, selectVarList, applyCut, varHasIndex, pickIndex
from prepareSequentialTraining import loadDataFrame as loadSequentialDataFrame

from collections import namedtuple
Sample = namedtuple('Sample', 'name' 'path')

inputDir = '/project/etp5/dhandl/samples/SUSY/Stop1L/FullRun2/hdf5/cut_mt30_met60_preselection_v60/'

SIGNAL = [['mc16a_bWN_650_500', 'mc16d_bWN_650_500', 'mc16e_bWN_650_500'], ['mc16a_bWN_650_560', 'mc16d_bWN_650_560', 'mc16e_bWN_650_560'], ['mc16a_bWN_700_550', 'mc16d_bWN_700_550', 'mc16e_bWN_700_550'],]
#SIGNAL = ['stop_tN_800_500']

BACKGROUND = [['mc16a_ttbar', 'mc16d_ttbar', 'mc16e_ttbar'], ['mc16a_singletop', 'mc16d_singletop', 'mc16e_singletop'], ['mc16a_wjets', 'mc16d_wjets', 'mc16e_wjets'], ['mc16a_ttV', 'mc16d_ttV', 'mc16e_ttV'], ['mc16a_multiboson', 'mc16d_multiboson', 'mc16e_multiboson']]

BWN_PRESEL_BKG = 10673.12
BWN_PRESEL_SIG = 397.53  # bWN_500_380
BWN_PRESEL_SIG = 169.34  # bWN_650_500
BWN_PRESEL_SIG = 52.37  # bWN_650_560
BWN_PRESEL_SIG = 115.87  # bWN_700_550


PRESELECTION = [
                {'name':'n_jet',  'threshold':4,      'type':'geq'},
                {'name':'n_bjet',  'threshold':1,      'type':'geq'},
                {'name':'met',    'threshold':400e3,  'type':'geq'},
                {'name':'mt',    'threshold':150e3,  'type':'geq'},
                {'name':'n_lep',  'threshold':1,      'type':'exact'},
                {'name':'dphi_met_lep',    'threshold':2.,  'type':'leq'},
                {'name':'m_bl',    'threshold':80e3,  'type':'leq'},
                {'name':'jet_pt[0]',    'threshold':300e3,  'type':'geq'},
                {'name':'bjet_pt[0]',    'threshold':80e3,  'type':'leq'},
                {'name':'lep_pt',  'threshold':25e3,      'type':'geq'}
               ]

WEIGHTS = [
           'weight',
           'lumi_weight',
           'xs_weight',
           'sf_total'
          ]

LUMI = 140e3

VAR = [ 
        'amt2',
      ]

def asimovZ(s, b, b_err, syst=False):
  tot = s + b
  b2 = b*b
  if syst:
    b_err2 = np.sqrt( b_err*b_err + (b*0.25)*(b*0.25) )
  else:
    b_err2 = b_err * b_err
  b_plus_err2 = b + b_err2
  Z = np.sqrt(2 * ((tot)*np.log(tot * b_plus_err2 / (b2 + tot * b_err2)) - b2 / b_err2 * np.log(1 + b_err2 * s / (b * b_plus_err2))))
  return Z


def pickBenchmark(signal, delimiter='_'):
  try:
    name = signal.split(delimiter)
    x = name[2]
    y = name[3]
    return x, y
  except Exception:
    print 'ERROR: No matching pattern for benchmark {}'.format(signal)
    return 0


def main():

  ###########################
  # Read and evaluate signals
  ###########################

  Signal = []
  for smp in SIGNAL:
    first = True
    for s in smp:
      print 'Sample:\t',s
      x, y = pickBenchmark(s)
      _df, _weight = loadDataFrame(os.path.join(inputDir, s+'/'), PRESELECTION, VAR, WEIGHTS, LUMI)
      print _df.shape,_weight.shape
      if first:
        df = _df.copy()
        weight = _weight.copy()
        first = False
      else: 
        df = pd.concat((df, _df), ignore_index=True)
        weight = pd.concat((weight, _weight), ignore_index=True)

    sigma = np.sum(weight.values**2.)
    
    Signal.append({'name':s[6:], 'm_stop':x, 'm_X':y, 'dataset':df, 'weight':weight, 'nEvents':weight.sum(), 'nSigma':np.sqrt(sigma)})

    del df, weight 

  ###########################
  # Read and evaluate backgrounds 
  ###########################
  
  totBkgEvents = 0.
  totBkgVar = 0.
  Background = []
  for smp in BACKGROUND:
    first = True
    for b in smp:
      print 'Sample:\t',b
      _df, _weight = loadDataFrame(os.path.join(inputDir, b+'/'), PRESELECTION, VAR, WEIGHTS, LUMI)
      print _df.shape,_weight.shape
      if first:
        df = _df.copy()
        weight = _weight.copy()
        first = False
      else:        
        df = pd.concat((df, _df), ignore_index=True)
        weight = pd.concat((weight, _weight), ignore_index=True)

    totBkgEvents += weight.sum()
    totBkgVar += np.sum(weight.values**2.)
    
    Background.append({'name':b, 'dataset':df, 'weight':weight, 'nEvents':weight.sum(), 'nSigma':np.sqrt(np.sum(weight.values**2.))})

    del df, weight
    
    total_rel_err = np.sqrt(totBkgVar/totBkgEvents**2. + (totBkgEvents*0.25)**2.) 

  print 'Bkg:\t%.2f +/- %.2f'%(totBkgEvents,np.sqrt(totBkgVar))

  for s in Signal:
    significance = []
    significance_err = []
    asimov = []
    asimov_err = []


    s['Z'] = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(s['nEvents'], totBkgEvents, total_rel_err)
    s['ams'] = asimovZ( s['nEvents'], totBkgEvents, np.sqrt(totBkgVar))
    print 'Z:\t%.2f'%s['Z']
    print 'Asimov:\t%.2f'%s['ams']
    print 'Sig %s:\t%.2f +/- %.2f'%(s['name'], s['nEvents'],s['nSigma'])
    print 'r_bkg:\t%.2f'%(1.-(totBkgEvents/BWN_PRESEL_BKG))
    print 'e_sig:\t%.2f'%((s['nEvents']/BWN_PRESEL_SIG))


if __name__ == "__main__":
    main()
   

