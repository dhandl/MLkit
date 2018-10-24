#!/usr/bin/env python

import os, sys
import numpy as np
import pandas as pd
import root_pandas as rp

from array import array
from keras.models import load_model
from sklearn.externals import joblib

import ROOT

VAR = [
        #'ht',
        #'jet_pt[0]',
        #'bjet_pt[0]',
        #'amt2',
        'met',
        'met_phi',
        'dphi_met_lep',
        'mt',
        #'dphi_b_lep_max',
        #'dphi_jet0_ptmiss',
        #'met_sig',
        #'met_proj_lep',
        #'ht_sig',
        #'m_bl',
        #'dr_bjet_lep',
        #'mT_blMET', #15vars
        'n_jet',
        'n_bjet',
        #'jet_pt[1]',
        #'jet_pt[2]',
        #'jet_pt[3]',
        #'lep_pt[0]',
        #'dr_jet_jet_max',
        #'ttbar_pt',
        #'ttbar_dphi',
        #'m_jet1_jet2',
        #'m_jet_jet_min',
        #'m_jet_jet_max'
        #'tt_cat_TRUTH3'
        #'n_lep',
        'lep_pt[0]',
        'lep_eta[0]',
        'lep_phi[0]',
        'lep_e[0]'
        #'jet_eta[0]',
        #'jet_phi[0]',
        #'dphi_jet1_ptmiss',
        #'dphi_jet2_ptmiss',
        #'dphi_jet3_ptmiss',
        #'dphi_min_ptmiss',
        #'dphi_b_ptmiss_max'
]

MODEL = 'TrainedModels/models/2018-09-28_00-38_RNN_jetOnly_ADAM_leayReLU_LSTM32_128NNlayer_batch32_NormalInitializer_l2-0p01.h5'

SCALER = 'TrainedModels/models/2018-09-28_00-38_RNN_jetOnly_ADAM_leayReLU_LSTM32_128NNlayer_batch32_NormalInitializer_l2-0p01_scaler.pkl'

#--------- important for RNN's ----------#
rnn = True
SEQ_SCALER = 'TrainedModels/datasets/20180927_stop_bWN_450_300TRUTH_ttbar_singletop_wjets_mt110_met230_scaling.json'
COLLECTION = ['jet'] 
REMOVE_VAR = ['_m', '_mv2c10', '_id', '0_pt', '0_eta', '0_phi', '0_e', '1_pt', '1_eta', '1_phi', '1_e']
#----------------------------------------#


def getVarValue(c, var):
  varNameHisto = var
  leaf = c.GetAlias(varNameHisto)
  if leaf!='':
    try:
      return c.GetLeaf(leaf).GetValue(n)
    except:
      raise Exception("Unsuccessful getVarValue for leaf %s and index %i"%(leaf, n))
  else:
    i = 0 
    if varHasIndex(var):
      i, name = pickIndex(var)
      l = c.GetLeaf(name)
    else:
      l = c.GetLeaf(var)
    if l:return l.GetValue(int(i))
    else:return float('nan')

 
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


# probably necessary for RNN's, 
def getObjDict(c, prefix, variables, i=0):
  variable = {}
  for var in variables:
    variable.update({prefix+var:c.GetLeaf(prefix+var).GetValue(i)})
  return variable


def getJets(c):
  addJetVars =  []
  if c=="branches":return ['n_jet','jet_pt','jet_eta', 'jet_phi', 'jet_e'] + ['jet_'+x for x in addJetVars]
  nJet = int(c.GetLeaf("n_jet").GetValue())
  jets=[]
  for i in range(nJet):
    jet = getObjDict(c, 'jet_', ['pt','eta', 'phi', 'e'], i)
    jet.update(getObjDict(c, 'jet_', addJetVars, i))
    jets.append(jet)
  return jets


def transformCollection(jets, name):
  df_jets = pd.DataFrame(index=range(0), columns=[name+'_pt', name+'_eta', name+'_phi', name+'_e'])

  pt=[p['jet_pt'] for p in jets]
  jet_pt = [{'jet_pt':pt}]
  jet_pt = pd.DataFrame(jet_pt)

  eta=[p['jet_eta'] for p in jets]
  jet_eta = [{'jet_eta':eta}]
  jet_eta = pd.DataFrame(jet_eta)

  phi=[p['jet_phi'] for p in jets]
  jet_phi = [{'jet_phi':phi}]
  jet_phi = pd.DataFrame(jet_phi)

  e=[p['jet_e'] for p in jets]
  jet_e = [{'jet_e':e}]
  jet_e = pd.DataFrame(jet_e)

  df_jets.loc[0] = [jet_pt.loc[0][0], jet_eta.loc[0][0],jet_phi.loc[0][0],jet_e.loc[0][0]]
  return {'df':df_jets, 'name':name, 'vars':[name+'_pt', name+'_eta', name+'_phi', name+'_e']}


def create_scale_stream(df, num_obj, sort_col, VAR_FILE_NAME):
  n_variables = df.shape[1]
  var_names = df.keys()
  data = np.zeros((df.shape[0], num_obj, n_variables), dtype='float32')
  
  # call functions to build X (a.k.a. data)
  sort_objects(df, data, sort_col, num_obj)
  Xobj = data
  
  scale(Xobj, var_names, VAR_FILE_NAME=VAR_FILE_NAME.replace('.h5','_scaling.json')) # apply scaling
  return Xobj


def sort_objects(df, data, SORT_COL, max_nobj):
  import tqdm
  # i = event number, event = all the variables for that event 
  for i, event in df.iterrows(): 
    # objs = [[pt's], [eta's], ...] of particles for each event

    #objs = np.array([v.tolist() for v in event.get_values()], dtype='float32')[:, (np.argsort(event[SORT_COL]))[::-1]]
    objs = np.array([v for v in event.get_values()], dtype='float32')[:, (np.argsort(event[SORT_COL]))[::-1]]
    # total number of tracks per jet      
    nobjs = objs.shape[1] 
    # take all tracks unless there are more than n_tracks 
    data[i, :(min(nobjs, max_nobj)), :] = objs.T[:(min(nobjs, max_nobj)), :] 
    # default value for missing tracks 
    data[i, (min(nobjs, max_nobj)):, :  ] = -999


def scale(data, var_names, VAR_FILE_NAME):
  import json
  scale = {}
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


def evaluate(model, dataset, scaler, seq_scaler=None, col=None, rnn=False):
  dataset = scaler.transform(dataset)

  if rnn:  
    for idx, c in enumerate(col):
      #c['n_max'] = max([len(j) for j in c['df'][c['name']+'_pt']])
      c['n_max'] = 15
      c['Xobj'] = create_scale_stream(c['df'], c['n_max'], sort_col=c['name']+'_pt', VAR_FILE_NAME=seq_scaler) 

    y_hat = model.predict([c['Xobj'] for c in col]+[dataset])

  else:
    y_hat = model.predict(dataset)

  return y_hat


#-----------------------------------------------#
#-----------------------------------------------#


def main():
  global VAR
  global MODEL
  global SCALER
  global SEQ_SCALER
  global COLLECTION
  global REMOVE_VAR

  if len(sys.argv) == 2:
    src = sys.argv[1:][0]
  #elif len(sys.argv) == 5:
  #  src, VAR, MODEL, SCALER = sys.argv[1:]
  elif len(sys.argv) == 4:
    src, MODEL, SCALER = sys.argv[1:]
  elif len(sys.argv) == 3:
    src, MODEL = sys.argv[1:]
  else:
    print 'Usage: evaluateFriendTree.py <sample directory> (<var list> <model file> <scaler file>)'
    return
 
  if not os.path.isdir(src):
    print 'No such folder "{}"'.format(src)

  scaler = joblib.load(SCALER)
  model = load_model(MODEL)

  # get all .root file in all subdirectories of <src>
  inFiles = [os.path.relpath(os.path.join(d,f), src) for (d, _, files) in os.walk(src) for f in files if f.endswith(".root")]

  print 'Going to evaluate model on following files:\n\t{files}'.format(files="\n- ".join(inFiles))

  for f in inFiles:
    fSrc = os.path.join(src,f)

    print "Evaluating '{}'...".format(f)

    inFile = ROOT.TFile(fSrc, "UPDATE")

    # Get all tress in this file
    for name in set([k.GetName() for k in inFile.GetListOfKeys() if k.GetClassName() == "TTree"]):
      print "\nDEBUG: Evaluate " + name
      t = inFile.Get(name)
    
      # define friend tree
      friend = ROOT.TTree(t.GetName()+"_ML", "")
      output = array('f', [0])
      friend.Branch('outputScore', output, 'outputScore/F')

      # loop over all events
      nevents = t.GetEntries()
      for i in range(nevents):
        t.GetEntry(i)
        if i%10000==0:
          print "Evaluated {} of {} total events".format(i, nevents)
        
        eventvars = []
        for var in VAR:
          eventvars.append(getVarValue(t, var))
        eventvars = np.array([eventvars])
        
        if rnn:
          collection = []
          jets = getJets(t)
          collection.append(transformCollection(jets, 'jet'))
          y_predict = evaluate(model, eventvars, scaler, SEQ_SCALER, rnn=True, col=collection)
        else:
          y_predict = evaluate(model, eventvars, scaler)
        #print y_predict[0,0], type(y_predict[0,0])
        output[0] = y_predict[:,0]

        # fill friend tree
        friend.Fill()

      # write friend tree to original file
      inFile.cd()
      friend.Write("", ROOT.TObject.kOverwrite)

      # make them friends <3, so we can use the output score in simple TTree::Draw() commands
      t.AddFriend(friend)
      t.Write(t.GetName(), ROOT.TObject.kOverwrite)


    # close files
    inFile.Close()
       
if __name__ == '__main__':
  main()
