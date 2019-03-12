#!/usr/bin/env python

import os, sys
import numpy as np
import pandas as pd
import root_numpy as rn

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
        'mt',
        'dphi_met_lep',
        'm_bl',
        'bjet_pt0',
        #'dphi_b_lep_max',
        #'dphi_jet0_ptmiss',
        #'met_sig',
        #'met_proj_lep',
        #'ht_sig',
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
        'lep_pt',      # see * ll.60
        'lep_eta',     # see * ll.60
        'lep_phi',     # see * ll.60
        'lep_e'        # see * ll.60
        #'jet_eta[0]',
        #'jet_phi[0]',
        #'dphi_jet1_ptmiss',
        #'dphi_jet2_ptmiss',
        #'dphi_jet3_ptmiss',
        #'dphi_min_ptmiss',
        #'dphi_b_ptmiss_max'
]

BRANCH_NAME = 'outputScore_RNN'

VAR_TRANSFORM = ['met', 'mt', 'lep_pt', 'lep_e', 'm_bl', 'bjet_pt0']

# *
# Index '[0]' not needed anymore, since array pollution is removed by function 'remove_array_pollution'.
# It is cleaner coding style to delete the index '[0]' ('lep_pt' instead of 'lep_pt[0]').
# If it is needed for whatever reason, the initialisation file of root_pandas "__init__.py" needs to be changed according to marked comments in that file. 


MODEL = '/project/etp5/dhandl/MachineLearning/finalModel/2018-12-21_11-20_RNN_jetOnly_ADAM_leakyReLU_LSTM32_128NNlayer_batch32_BatchNorm_NormalInitializer_l2-0p01.h5'
#MODEL = '/project/etp5/dhandl/MachineLearning/finalModel/2018-09-28_00-38_RNN_jetOnly_ADAM_leayReLU_LSTM32_128NNlayer_batch32_NormalInitializer_l2-0p01.h5'

SCALER = '/project/etp5/dhandl/MachineLearning/finalModel/2018-12-21_11-20_RNN_jetOnly_ADAM_leakyReLU_LSTM32_128NNlayer_batch32_BatchNorm_NormalInitializer_l2-0p01_scaler.pkl'
#SCALER = '/project/etp5/dhandl/MachineLearning/finalModel/2018-09-28_00-38_RNN_jetOnly_ADAM_leayReLU_LSTM32_128NNlayer_batch32_NormalInitializer_l2-0p01_scaler.pkl'

CHUNKSIZE = 100000

#--------- important for RNN's ----------#
rnn = True
SEQ_SCALER = '/project/etp5/dhandl/MachineLearning/finalModel/20181220_stop_bWN_450_300TRUTH_allBkgs_njet4_nbjet1_met230_mt110_RNN_scaling.json'
# previous model
#SEQ_SCALER = '/project/etp5/dhandl/MachineLearning/finalModel/20180927_stop_bWN_450_300TRUTH_ttbar_singletop_wjets_mt110_met230_scaling.json'
COLLECTION = ['jet'] 
COLLECTION_VAR = ['_pt', '_eta', '_phi', '_e']
COLLECTION_TRANSFORM = ['_pt','_e']
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


# Probably necessary for RNN's, 
def getObjDict(c, prefix, variables, i=0):
  variable = {}
  for var in variables:
    variable.update({prefix+var:c.GetLeaf(prefix+var).GetValue(i)})
  return variable


def getJets(c):
  addJetVars =  []
  if c=="branches":return ['n_jet','jet_pt','jet_eta', 'jet_phi', 'jet_m'] + ['jet_'+x for x in addJetVars]
  #if c=="branches":return ['n_jet','jet_pt','jet_eta', 'jet_phi', 'jet_e'] + ['jet_'+x for x in addJetVars]
  nJet = int(c.GetLeaf("n_jet").GetValue())
  jets=[]
  for i in range(nJet):
    jet = getObjDict(c, 'jet_', ['pt','eta', 'phi', 'm'], i)
    #jet = getObjDict(c, 'jet_', ['pt','eta', 'phi', 'e'], i)
    jet.update(getObjDict(c, 'jet_', addJetVars, i))
    e = ROOT.TLorentzVector()
    e.SetPtEtaPhiM(jet['jet_pt'], jet['jet_eta'], jet['jet_phi'], jet['jet_m'])
    jet.update({"jet_e":e.E()})
 
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
  data = np.zeros((df.shape[0], num_obj, n_variables), dtype='float64')
  
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
    objs = np.array([v for v in event.get_values()], dtype='float64')[:, (np.argsort(event[SORT_COL]))[::-1]]
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
    data[:, :, v][f != -999] = slc.astype('float64')


def evaluate(model, dataset, scaler, seq_scaler=None, col=None, rnn=False):
  dataset = scaler.transform(dataset)

  if rnn:  
    for idx, c in enumerate(col):
      #c['n_max'] = max([len(j) for j in c['df'][c['name']+'_pt']])
      c['n_max'] = 16
      c['Xobj'] = create_scale_stream(c['df'], c['n_max'], sort_col=c['name']+'_pt', VAR_FILE_NAME=seq_scaler) 
    y_hat = model.predict([c['Xobj'] for c in col]+[dataset])

  else:
    y_hat = model.predict(dataset)

  return y_hat


def multiply_by_thousand(unscaled_dataframe, variable_index):
  """ Some frameworks use different units as other frameworks, whyever. In order to be able to compare the data anyway,
      all units are scaled to MeV. In this function we assume, that the used unit is GeV. So, in order to get the used
      GeV to MeV one needs to multiply the value by thousand. This function takes as arguments the dataframe which
      should be scaled and a list of variables which use GeV as unit and returns a dataframe in terms of MeV."""
  
  # Loop through each element of variable index and multiply by thousand 
  for ele in variable_index:
    unscaled_dataframe[:,ele] = unscaled_dataframe[:,ele] * 1000

  # Nb. unscaled_dataframe is scaled now.  
  
  return unscaled_dataframe


def multiply_jets_by_thousand(complicated_list, variables_list):
  """ Same problem as stated in function 'multiply_by_thousand'. If this function is called, jet energies will be given in terms of GeV.
      However, this model needs values of MeV. This function does that job.
      First we read the pandas DataFrame from the complicated list 'jet_seq'. For each variable we then create a list containing the entries.
      I.e., list_raw=[entry1, entry2, entry3,...]. In this case however, the entries are lists of their own (hence the name list_arrays).
      Each entry belongs to an own event. We then create another list, which will contain the scaled values. For each entry of list_raw we
      create the scaled entry, type of list, and append that scaled entry to scaled list of arrays.
      As a last step we swap the pandas Series with the newl created pandas Series. After transforming it back to the complicated list, we return mentioned."""

  # Read pandas DataFrame from complicated list jet_seq
  jet_df = complicated_list[0]['df']
  
  # We iterate through each variable we want to scale  
  for var in variables_list:
    # Create a new list, which contains values of pandas Series of column var
    # Since its entries are lists, we basically have a list of list
    list_arrays = jet_df[var].tolist()
    
    # Initialise empty list
    scaled_list_arrays = []

    # Loop through each entry of list...
    for lst in list_arrays:
      
      # ... and basically create a copy of values times 1000 in new list
      scaled_list_arrays.append([value*1000 for value in lst])

    # Update series of old values with series created out of new list
    jet_df[var].update(pd.Series(scaled_list_arrays))
  
   #Rewrite in format of complicated list
  complicated_list[0]['df'] = jet_df

  return complicated_list


def remove_array_pollution(dirtydf, variables_list):
  """ It is seen that the lepton variables are of type array containing only one entry, which is
      the wished value. This function takes the name of the variable, of which column shall be
      transformed, and returns a new dataframe with the same data without that nuisance array. 
      It returns the same dataframe with the values of given variable name changed from arrays into
      values of that array."""

  for var in variables_list:
    # Creating a dictionary out of the column of given variable
    # Nb. Form of dict_varname due to function "to_dict" with argument 'records'
    # to_dict('records') returns [{'col1': 1.0, 'col2': 0.5}, {'col1': 2.0, 'col2': 0.75}]
    dict_varname = dirtydf.to_dict('records')
  
    # Temporary list in which values will be written in  
    temp_list = []    

    # Loops through each entry of that dictionary (as upperly defined) and searches for the value of the key "variable_name" (because everyother variable is stored too)
    # Of that value, which is an array, it takes the zeroth value and appends that into a list. 
    for ele in dict_varname:
      try:
        temp_list.append(ele[var][0])
      except IndexError:
        #ele[var].append(np.nan)
        temp_list.append(np.nan)
  
    # Creating a new dictionary in preperation for new dataframe
    dict_df = {var: temp_list}
  
    # Create new DataFrame with said dictionary
    temp_dframe = pd.DataFrame(dict_df, dtype=np.float64)
  
    # Overwrite column of dirty dataframe with fresh dataframe
    dirtydf[var] = temp_dframe[var]
    # Nb. dirtydf is now clean.
  
  return dirtydf          


#-----------------------------------------------#
#-----------------------------------------------#


def main():
  global VAR
  global BRANCH_NAME 
  global MODEL
  global SCALER
  global SEQ_SCALER
  global COLLECTION
  global CHUNKSIZE

  # Check number of arguments and act respectively thereof
  if len(sys.argv) == 2:
    src = sys.argv[1:][0]
  elif len(sys.argv) == 5:
    src, VAR, MODEL, SCALER = sys.argv[1:]
  elif len(sys.argv) == 4:
    src, MODEL, SCALER = sys.argv[1:]
  elif len(sys.argv) == 3:
    src, MODEL = sys.argv[1:]
  else:
    print 'Usage: evaluateFriendTree_truth.py <sample directory> (<var list> <model file> <scaler file>)'
    return
 
  if not os.path.isdir(src):
    print 'No such folder "{}"'.format(src)

  # Load scaler and model and define chunksize
  scaler = joblib.load(SCALER)
  model = load_model(MODEL)
  
  # Get all .root file in all subdirectories of <src>
  inFiles = [os.path.relpath(os.path.join(d,f), src) for (d, _, files) in os.walk(src) for f in files if f.endswith(".root")]

  print '\nGoing to evaluate model on following files:\n\t{files}\n'.format(files="\n- ".join(inFiles))

  # Loop through all files
  for f in inFiles:
  
    # Creating complete source for each file    
    fSrc = os.path.join(src,f)

    print "Evaluating '{}'...".format(f)

    # Open current file in list of files "inFiles" with complete source path
    inFile = ROOT.TFile(fSrc)

    # Get all trees in this file
    for name in set([k.GetName() for k in inFile.GetListOfKeys() if k.GetClassName() == "TTree"]):

      t = inFile.Get(name)
      # Defining and getting strings of tree names      
      name_data_tree = name
      #name_new_tree = t.GetName()+"_ML"

      branches = [b.GetName() for b in t.GetListOfBranches()]
      #if ("_ML" in name) or ("_lumi" in name):
      inList = filter(lambda x: x == BRANCH_NAME, branches)
      if len(inList) > 0:
        print "Skipping {} ...".format(name)
        continue

      # Loop over all events
      nevents = t.GetEntries()
      
      # Initialising indices   
      i = 0
      idx = 0
      
      # Print information
      print 'This file has {} entries and chunks of size {} are used, thus resulting in {} loops.\n'.format(nevents, CHUNKSIZE, nevents//CHUNKSIZE + 1 )    
    
      # Loop over every chunk
      #for chunks in rp.read_root(fSrc, chunksize=CHUNKSIZE, columns=VAR):
      for start in range(0, nevents, CHUNKSIZE):
        # Print status
        print 'Processing loop {} out of {}'.format(idx+1, nevents//CHUNKSIZE + 1)

        chunk = rn.tree2array(t, branches=VAR, start=start, stop=start+CHUNKSIZE)
        X_eval = rn.rec2array(chunk[VAR]) 
        #X_eval = pd.DataFrame(chunk) 
        
        # Remove array polution and transform entry of type array in entry of type float
        # Transforms [458.54823] into 458.54823
        # Possible loss of accuracy here due to rounding
        #print '\nDEBUG: BEFORE\t',X_eval[:5]
        #X_eval = remove_array_pollution(X_eval, ['bjet_pt', 'lep_pt', 'lep_eta', 'lep_phi', 'lep_e'])
        #print '\nDEBUG: AFTER\t',X_eval[:5]

        # Define a sequence of jets for each chunk of entries.
        # Each sequence(list) contains as entries all dataframe of events in that chunk. 
        if rnn: 
          # Print status
          print '\tSetting jet events to right format.'
          jet_chunk = rn.tree2array(t, branches=['jet_pt', 'jet_eta', 'jet_phi', 'jet_e'], start=start, stop=start+CHUNKSIZE)
          jet_seq = [{'df':pd.DataFrame(jet_chunk), 'name':'jet', 'vars':['jet_pt', 'jet_eta', 'jet_phi', 'jet_e']}]


        #for i in range(X_eval.shape[0]):          

        #  t.GetEntry(i + start*X_eval.shape[0])
        #  
        #  if rnn:
        #    collection = []
        #    jets = getJets(t)
        #    collection.append(transformCollection(jets, 'jet'))
        #    jet_seq[0]['df'] = pd.concat((jet_seq[0]['df'], collection[0]['df']), axis=0, ignore_index=True)

        var_indices = [VAR.index(v) for v in VAR_TRANSFORM]
        X_eval = multiply_by_thousand(X_eval, var_indices)
        jet_seq = multiply_jets_by_thousand(jet_seq, ['jet_pt', 'jet_e'])

        # Actual evaluation with status printing
        print '\tEvaluating chunk of data.'
        if rnn:
          y_predict = evaluate(model, X_eval, scaler, SEQ_SCALER, rnn=True, col=jet_seq)
        else:
          y_predict = evaluate(model, X_eval, scaler)
        
        # Take only the first column of the output
        output = y_predict[:,0]
        friend_df = pd.DataFrame(np.array(output, dtype=[(BRANCH_NAME, np.float64)]))
        friend_tree = friend_df.to_records()[[BRANCH_NAME]]
        #if start == 0:
        #  mode = 'recreate'
        #else:
        mode = 'update'
        print "Write to file"
        # Write to new root file
        rn.array2root(friend_tree, fSrc, treename=name_data_tree, mode=mode)
        print "Done"

        # Increasing chunk index
        idx = idx + 1

    # Close file
    inFile.Close()

    # Now that data processing is over and file is closed.


    # Now file will be opened in new instance and the trees are going to be befriended
    
    #Open ROOT file again       
    #root_file = ROOT.TFile(fSrc, "UPDATE")
    
    # Get outputScore tree    
    #outputScore_tree = root_file.Get(name_new_tree)
    
    # Write outpurScore tree
    #root_file.cd()
    #outputScore_tree.Write("", ROOT.TObject.kOverwrite)

    # Get tree with original data 
    #data_tree = root_file.Get(name_data_tree)

    # Make them friends <3
    #data_tree.AddFriend(name_new_tree)
    
    # Write data tree and close file  
    #data_tree.Write(data_tree.GetName(), ROOT.TObject.kOverwrite)       
    #root_file.Close()
    

if __name__ == '__main__':
  main()
