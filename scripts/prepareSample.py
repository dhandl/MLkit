#!/usr/bin/env python

import ROOT
import os
import sys
import pandas as pd
import h5py
from root2pandas import root2pandas

#CUT = "(dphi_jet0_ptmiss > 0.4) && (dphi_jet1_ptmiss > 0.4) && (n_jet>=4) && (n_bjet>0) && (jet_pt[0]>50e3) && (jet_pt[1]>25e3) && (jet_pt[2]>25e3) && (jet_pt[3]>25e3) && (mt>30e3) && !((mT2tauLooseTau_GeV > -0.5) && (mT2tauLooseTau_GeV < 80)) && (met >120e3)"
CUT ="(1)"

variables = [
              "n_jet", "jet_pt", "jet_eta", "jet_phi", "jet_e", "jet_deltaRj",
#              "n_bjet", "bjet_pt", "bjet_eta", "bjet_phi", "bjet_e",
              "lep_pt", "lep_eta", "lep_phi", "lep_e",
              "met",
              "dphi_jet0_ptmiss", "dphi_jet1_ptmiss", "dphi_jet2_ptmiss", "dphi_jet3_ptmiss",
              "dphi_min_ptmiss",
              "dphi_met_lep",
              "dr_jet_jet_min", "dr_jet_jet_max", "dr_lep_jet_min", "dr_lep_jet_max",
              "dphi_jet_jet_min", "dphi_jet_jet_max", "dphi_lep_jet_min", "dphi_lep_jet_max",
              "deta_jet_jet_min", "deta_jet_jet_max", "deta_lep_jet_min", "deta_lep_jet_max",
              "m_jet1_jet2", "m_jet_jet_min", "m_jet_jet_max",
              "event_number", "run_number", "lumi_block", "mc_channel_number", "bcid",
              "weight",
              "xs_weight"
]

def filesize(num):
  for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
    if abs(num) < 1024.0:
      return "%3.1f%sB" % (num, unit)
    num /= 1024.0
  return "%.1fYiB" % num

def main():
  global CUT
  if not len(sys.argv) == 3:
    print "Usage: cut-tree.py <source directory> <destination directory>"
    return
  elif len(sys.argv) == 4:
    src, dest, CUT = sys.argv[1:]
  else:
    src, dest = sys.argv[1:]

  if not os.path.isdir(src):
    print "No such folder '{}'".format(src)
    return

  if not os.path.isdir(dest):
    print "No such folder '{}'".format(dest)
    return

  # get all .root files in all subdirectories of <src>
  inFiles = [os.path.relpath(os.path.join(d, f), src) for (d, _, files) in os.walk(src) for f in files if f.endswith("root")] 

  print "Going to preprocess following files\nfrom\n\t{src}\nto\n\t{dest}\n(without overwriting existing files)\n\n- {files}" \
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

    #print "Copying '{}'...".format(f),

    f = ROOT.TFile(fSrc)

    # Get all trees in this file
    for name in set([k.GetName() for k in f.GetListOfKeys() if k.GetClassName() == "TTree"]):
      print "\nDEBUG: Processing " + name
      t = f.Get(name)

      # Resetting all branches and only set the variables from list variables
      t.SetBranchStatus("*", 0)
      for var in variables:
        t.SetBranchStatus(var, 1)

      t.Draw(">>eList",CUT) #Get the event list 'eList' which has all the events satisfying the cut
      elist = ROOT.gDirectory.Get("eList")
      nevents = elist.GetN()
      print "\nDEBUG: Will loop over {} events in file {} ".format(nevents, fSrc)
      first = True
      last = nevents-1
      for i in range(nevents):
        if i%10000==0:
          print i
        if first:
          fDest2 =fDest.replace(".root","_"+str(i/1000000)+".root")
          fCopy = ROOT.TFile(fDest2, "RECREATE")
          fCopy.cd()
          tCopy = t.CloneTree(0)
          first = False
        if i%1000000==0 and i!=0:
          #fCopy.Write()
          fCopy.Close()
          fDest2 =fDest.replace(".root","_"+str(i/1000000)+".root")
          fCopy = ROOT.TFile(fDest2, "RECREATE")
          fCopy.cd()
          tCopy = t.CloneTree(0)
        t.GetEntry(i)
        tCopy.Fill()
        if i == last:
          print "File {} processed!".format(fSrc)
          fCopy.Write()
          #fCopy.Close()
      # Open destination file for this tree. This is important as otherwise the tree would get written to
      # memory by default when doing CopyTree
      #fCopy = ROOT.TFile(fDest, "RECREATE")
      #fCopy.cd()

      #tCopy = t.CopyTree(CUT)
      #tCopy.AutoSave()
      #fCopy.Close()
      
    f.Close()
  # Create data frame from preprocessed .root file
  output = os.listdir(dest)
  for ofile in output:
    fRoot = os.path.join(dest, ofile) 

    f = ROOT.TFile(fRoot)

    # Get all trees in this file
    for name in set([k.GetName() for k in f.GetListOfKeys() if k.GetClassName() == "TTree"]):
      print "\nDEBUG: Processing " + name
      t = f.Get(name)
      outFile = fRoot.replace(".root", ".h5")
      df = root2pandas(fRoot, name)

      # save a pandas df to hdf5 (better to first convert it back to ndarray, to be fair)
      store = pd.HDFStore(os.path.join(dest, outFile), "w")
      store.put(name, df, data_columns=df.columns)
      store.close()

      # let's load it back in to make sure it actually worked!
      new_df = pd.read_hdf(os.path.join(dest, outFile), name)
      # -- check the shape again -- nice check to run every time you create a df
      print "File check! {}".format(os.path.join(dest, outFile))
      print "(Number of events, Number of branches): ",new_df.shape
    #print "OK Saved {}".format(filesize(os.stat(fSrc).st_size - os.stat(fDest).st_size))

  writeInfo = os.path.join(dest, "info.txt")
  if os.path.exists(writeInfo):
    while True:
      i = raw_input("'{}' exists. Should it be overwritten? (y|n) ".format(writeInfo)).strip().lower()
      if i == "y":
        break
      elif i == "n":
        return

  with open(writeInfo, "w") as f:
    f.write("Preprocess Info!\nCut: {}".format(CUT))
    f.write("\nVariables:")
    for var in variables:
      f.write("\n"+var)

if __name__ == "__main__":
  main()

