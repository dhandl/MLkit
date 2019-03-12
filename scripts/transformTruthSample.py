#!/usr/bin/env python
import os, sys 
import subprocess
import pandas as pd
import tqdm
import ROOT

def transformHDF(path):

  df = pd.read_hdf(path)
  df['weight'] = df['eventWeight']
  #df['xs_weight'] = 5.985e-07 # stop xsec
  df['sf_total'] = 1.
  df['weight_sherpa22_njets'] = 1.

  df['met'] = df['met']*1000.
  df['m_bl'] = df['m_bl']*1000.
  df['met_x'] = df['met_x']*1000.
  df['met_y'] = df['met_y']*1000.
  df['met_perp'] = df['met_perp']*1000.
  df['mt'] = df['mt']*1000.
  df['ht'] = df['ht']*1000.
  df['jet_pt'] = df['jet_pt']*1000.
  df['jet_m'] = df['jet_m']*1000.

  df['lep_pt'] = df['lep_pt']*1000.
  df['lep_e'] = df['lep_e']*1000.
  df['lep_pt'] = df[['lep_pt']].values.tolist()
  df['lep_eta'] = df[['lep_eta']].values.tolist()
  df['lep_phi'] = df[['lep_phi']].values.tolist()
  df['lep_e'] = df[['lep_e']].values.tolist()

  df['bjet_pt0'] = df['bjet_pt0']*1000.
  df['bjet_e0'] = df['bjet_e0']*1000.
  df['bjet_pt1'] = df['bjet_pt1']*1000.
  df['bjet_e1'] = df['bjet_e1']*1000.
  df['bjet_pt'] = df[['bjet_pt0', 'bjet_pt1']].values.tolist()
  df['bjet_eta'] = df[['bjet_eta0', 'bjet_eta1']].values.tolist()
  df['bjet_phi'] = df[['bjet_phi0', 'bjet_phi1']].values.tolist()
  df['bjet_e'] = df[['bjet_e0', 'bjet_e1']].values.tolist() 

  df_jet = pd.DataFrame(index=range(df.shape[0]), columns=['jet_e'])

  for i, event in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
    energy = []
    for jet in range(event['n_jet']):
      e = ROOT.TLorentzVector()
      e.SetPtEtaPhiM(event['jet_pt'][jet], event['jet_eta'][jet], event['jet_phi'][jet], event['jet_m'][jet])
      energy.append(e.E())
    jet_e = [{"jet_e":energy}]
    jet_e = pd.DataFrame(jet_e)
    df_jet.loc[i] = [jet_e.loc[0][0]] 
  df = pd.concat((df, df_jet), axis=1) 
  
  df.to_hdf(path, key='df', mode='w')
  print '\tTransformed'

def main():
  if len(sys.argv) == 2:
    src = sys.argv[1:][0]
  else:
    print 'Usage: transformTruthSample.py <SAMPLE DIRECTORY>'

  if not os.path.isdir(src):
    print 'No such directory "{}"'.format(src)

  inFiles = [os.path.relpath(os.path.join(d,f), src) for (d, _, files) in os.walk(src) for f in files if f.endswith(".h5")]

  print '\nGoing to transform following TRUTH samples:\n\t{files}\n'.format(files="\n- ".join(inFiles))

  for f in inFiles:
    transformHDF(f)

if __name__ == '__main__':
  main()
