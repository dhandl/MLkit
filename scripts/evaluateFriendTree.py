#!/usr/bin/env python

import os, sys
import numpy as np

from array import array
from keras.models import load_model
from sklearn.externals import joblib

import ROOT

# load model, scaler and trained variables
# get input .root files
# start eventloop
# grab input vars and fill them in array
# predict output and write to tree
# add friend tree

VAR = [
        #'ht',
        'jet_pt[0]',
        'bjet_pt[0]',
        'amt2',
        'mt',
        'met',
        #'met_phi',
        'dphi_met_lep',
        'dphi_b_lep_max',
        'dphi_jet0_ptmiss',
        #'met_sig',
        'met_proj_lep',
        'ht_sig',
        'm_bl',
        'dr_bjet_lep',
        'mT_blMET', #15vars
        'n_jet',
        'n_bjet'
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
        #'lep_pt',
        #'lep_eta',
        #'lep_phi',
        #'lep_e'
        #'jet_eta[0]',
        #'jet_phi[0]',
        #'dphi_jet1_ptmiss',
        #'dphi_jet2_ptmiss',
        #'dphi_jet3_ptmiss',
        #'dphi_min_ptmiss',
        #'dphi_b_ptmiss_max'
]

MODEL = ''

SCALER = ''

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
    variable.update({var:c.GetLeaf(prefix+var).GetValue(i)})
  return variable

def getJets(c):
  addJetVars =  ['e', 'mv2c10', 'loosebad', 'tightbad', 'Jvt', 'truthLabel', 'HadronConeExclTruthLabelID', 'killedByPhoton']
  if c=="branches":return ['n_jet','jet_pt','jet_eta', 'jet_phi'] + ['jet_'+x for x in addJetVars]
  nJet = int(getVarValue(c, 'n_jet'))
  jets=[]
  for i in range(nJet):
    jet = getObjDict(c, 'jet_', ['pt','eta', 'phi'], i)
    jet.update(getObjDict(c, 'jet_', addJetVars, i))
    jets.append(jet)
  return jets

def evaluate(model, dataset, scaler):
  dataset = scaler.transform(dataset)
  y_hat = model.predict(dataset)
  return y_hat

def main():
  global VAR
  global MODEL
  global SCALER

  if len(sys.argv) == 2:
    src = sys.argv[1:]
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
        
        y_predict = evaluate(model, eventvars, scaler)
        print y_predict[0,0], type(y_predict[0,0])
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
