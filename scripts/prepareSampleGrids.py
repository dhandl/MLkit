#!/usr/bin/env python

import ROOT
import os
import sys
import pandas as pd
import numpy as np
import h5py
from root2pandas import root2pandas

variables = [
              'n_jet', 'jet_pt', 'jet_eta', 'jet_phi',
              'n_lep', 'lep_pt', 'lep_eta', 'lep_phi',
              'met', 'met_phi'
]

extraVars = [
              "stxe_trigger", "el_trigger", "mu_trigger", "lep_trig_req",
              "event_number", "run_number", "lumi_block", "mc_channel_number", "bcid"
]

weights = ['weight', 'xs_weight', 'sf_total', 'weight_sherpa22_njets']


def main():
  if not len(sys.argv) == 3:
    print "Usage: prepareSampleGrids.py <source directory> <destination directory>"
    return
  else:
    src, dest = sys.argv[1:]

  if not os.path.isdir(src):
    print "No such folder '{}'".format(src)
    return

  if not os.path.isdir(dest):
    print "No such folder '{}'".format(dest)
    return

  # get all .h5 files in all subdirectories of <src>
  inFiles = [os.path.relpath(os.path.join(d, f), src) for (d, _, files) in os.walk(src) for f in files if f.endswith("h5")] 

  print "Going to create event grids for following files\nfrom\n\t{src}\nto\n\t{dest}\n(without overwriting existing files)\n\n- {files}" \
    .format(src=src, dest=dest, files="\n- ".join(inFiles))

  while True:
    i = raw_input("Are you okay with that? (y|n) ").strip().lower()
    if i == "y":
      break
    elif i == "n":
      return

  for infile in inFiles:
    fSrc, fDest = os.path.join(src, infile), os.path.join(dest, infile)

    if not os.path.exists(os.path.dirname(fDest)):
      os.makedirs(os.path.dirname(fDest))

    if os.path.exists(fDest):
      print "Skipping " + infile
      continue

    f = pd.read_hdf(fSrc)

    for var in variables:
      found = False
      for key in f.keys():
        if var == key:
          found = True
          break
      if not found:
        print 'Variable {} not found in data frame! Data frame only contains following variables {}'.format(var,f.keys())
        return 0 

    # Create 2d grid from .h5 file
  
    ix = range(f.shape[0]) 
    weight_df = f[weights]
    extra_df = f[extraVars] 
    df = f[variables]
    
    grids = []
  
    xBins = np.arange(-3.2, 3.2+0.2, 0.2) #phi
    yBins = np.arange(-3, 3+0.2, 0.2) #eta
  
    nxPix = len(xBins)-1
    nyPix = len(yBins)-1
  
    for index, evt in df.iterrows():
      grid = np.zeros((nyPix, nxPix)) #Attention array is organized in (rows, cols) = (y,x)

      # met
      met_xbin = np.digitize(evt.met_phi, xBins)-1
      met_ybin = np.digitize(0., yBins)-1
      met_z = evt.met
      if grid.shape[1] > met_xbin and grid.shape[0] > met_ybin:
        grid[met_ybin][met_xbin] = met_z    

      # lepton
      lep_xbin = np.digitize(evt.lep_phi[0], xBins)-1
      lep_ybin = np.digitize(evt.lep_eta[0], yBins)-1
      lep_z = evt.lep_pt[0]
      if grid.shape[1] > lep_xbin and grid.shape[0] > lep_ybin:
        grid[lep_ybin][lep_xbin] = lep_z    
  
      # jets
      for i in range(evt.n_jet):
        jet_xbin = np.digitize(evt.jet_phi[i], xBins)-1 #phi
        jet_ybin = np.digitize(evt.jet_eta[i], yBins)-1 #eta
        jet_z = evt.jet_pt[i]
        if grid.shape[1] > lep_xbin and grid.shape[0] > lep_ybin:
          grid[jet_ybin][jet_xbin] = jet_z
  
      grids.append(grid)
    grids = np.array(grids)

    fDest = fDest.replace('.h5', '_eventGrids.h5')
    h5f = h5py.File(fDest, 'w')
    h5f.create_dataset('ix', data=ix)
    h5f.create_dataset('eventGrid', data=grids)
    h5f.create_dataset('weight', data=weight_df)
    h5f.create_dataset('addInfo', data=extra_df)

    h5f.close()

if __name__ == "__main__":
  main()


def deltaPhi(phi1, phi2):
  dphi = phi2-phi1
  if  dphi > np.pi:
    dphi -= 2.0*np.pi
  if dphi <= -np.pi:
    dphi += 2.0*np.pi
  return dphi
