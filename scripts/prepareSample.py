#!/usr/bin/env python

import os
import sys
import pandas as pd
import numpy as np
import root_numpy as rn
import ROOT

CUT = "(dphi_jet0_ptmiss > 0.4) && (dphi_jet1_ptmiss > 0.4) && !((mT2tauLooseTau_GeV > -0.5) && (mT2tauLooseTau_GeV < 80)) && (n_lep==1) && (lep_pt[0]>25e3) && (n_jet>=4) && (n_bjet>=1) && (jet_pt[0]>25e3) && (jet_pt[1]>25e3) && (jet_pt[2]>25e3) && (jet_pt[3]>25e3) && (mt>30e3) && (met>60e3)"

# cut for truth sample
#CUT = "(dphi_jet0_ptmiss > 0.4) && (dphi_jet1_ptmiss > 0.4) && !((mt2_tau > -0.5) && (mt2_tau < 80)) && (n_jet>=4) && (n_bjet>=1) && (n_lep==1) && (n_lep_hard==1) && (lep_pt>25) && (jet_pt[0]>25) && (jet_pt[1]>25) && (jet_pt[2]>25) && (jet_pt[3]>25) && (mt>30) && (met>60)"
#CUT ="(1)"

# variable list in SUSY5 tuples
VAR = [
              "n_jet", "jet_pt", "jet_eta", "jet_phi", "jet_e", "jet_mv2c10", 
              "n_bjet", "bjet_pt", "bjet_eta", "bjet_phi", "bjet_e",
              "n_lep", "lep_pt", "lep_eta", "lep_phi", "lep_e", "lep_charge",
              #"n_hadtop_cand", "hadtop_cand_pt", "hadtop_cand_eta", "hadtop_cand_phi", "hadtop_cand_m",
              #"n_hadw_cand", "hadw_cand_pt", "hadw_cand_eta", "hadw_cand_phi", "hadw_cand_m",
              "met", "met_phi", "mt", "ht", "met_sig", "ht_sig", "amt2", "met_perp", "lepPt_over_met", "mT2tauLooseTau_GeV",
              "dphi_jet0_ptmiss", "dphi_jet1_ptmiss", "dphi_jet2_ptmiss", "dphi_jet3_ptmiss",
              #"dphi_min_ptmiss",
              "dphi_met_lep", "m_bl",
              #"dphi_b_lep_max", "dphi_b_ptmiss_max", "met_proj_lep", "dr_bjet_lep", "m_bl", "mT_blMET",
              #"m_top_chi2",
              #"dr_jet_jet_min", "dr_jet_jet_max", "dr_lep_jet_min", "dr_lep_jet_max",
              #"dphi_jet_jet_min", "dphi_jet_jet_max", "dphi_lep_jet_min", "dphi_lep_jet_max",
              #"deta_jet_jet_min", "deta_jet_jet_max", "deta_lep_jet_min", "deta_lep_jet_max",
              #"ttbar_m", "ttbar_pt", "dphi_ttbar", "dphi_leptop_met", "dphi_hadtop_met",
              #"m_jet1_jet2", "m_jet_jet_min", "m_jet_jet_max",
              #"RJR_RISR", "RJR_MS", "RJR_PTISR", "RJR_nBJet",
              "stxe_trigger",# "el_trigger", "mu_trigger", "lep_trig_req",
              "event_number", "run_number", "lumi_block", "mc_channel_number", "bcid",
              "tt_cat",
              "weight",
              "lumi_weight",
              "xs_weight",
              "sf_total"
              #"weight_sherpa22_njets"
              #"Event",
              #"eventWeight"
]

# variable list in TRUTH3 tuple
#VAR = [
#              "n_jet", #"jet0_pt", "jet0_eta", "jet0_phi", "jet0_e",
#              "jet_pt", "jet_eta", "jet_phi", "jet_m", "jet_id",
#              #"jet1_pt", "jet1_eta", "jet1_phi", "jet1_e",
#              #"jet2_pt", "jet2_eta", "jet2_phi", "jet2_e",
#              #"jet3_pt", "jet3_eta", "jet3_phi", "jet3_e",
#              "n_bjet", "bjet_pt0", "bjet_eta0", "bjet_phi0", "bjet_e0",
#              "bjet_pt1", "bjet_eta1", "bjet_phi1", "bjet_e1",
#              "n_lep", "n_lep_hard", "n_lep_soft", "lep_pt", "lep_eta", "lep_phi", "lep_e",
#              #"n_hadtop_cand", "hadtop_cand_pt", "hadtop_cand_eta", "hadtop_cand_phi", "hadtop_cand_m",
#              #"n_hadw_cand", "hadw_cand_pt", "hadw_cand_eta", "hadw_cand_phi", "hadw_cand_m",
#              "met", "met_phi", "met_x", "met_y", "mt", "ht", "ht_sig", "amt2", "mt2_tau", "met_perp", "hadw_cand_m", "lepPt_over_met", "pt_W", "dphi_ttbar", "hadtop_cand_m0",
#              "dphi_jet0_ptmiss", "dphi_jet1_ptmiss",# "dphi_jet2_ptmiss", "dphi_jet3_ptmiss",
#              #"dphi_min_ptmiss",
#              "dphi_met_lep", "dr_bjet_lep", "minDphiMetBjet", "m_bl",
#              #"dphi_b_lep_max", "dphi_b_ptmiss_max", "met_proj_lep", "dr_bjet_lep" "mT_blMET",
#              "m_top_chi2",
#              #"dr_jet_jet_min", "dr_jet_jet_max", "dr_lep_jet_min", "dr_lep_jet_max",
#              #"dphi_jet_jet_min", "dphi_jet_jet_max", "dphi_lep_jet_min", "dphi_lep_jet_max",
#              #"deta_jet_jet_min", "deta_jet_jet_max", "deta_lep_jet_min", "deta_lep_jet_max",
#              #"ttbar_m", "ttbar_pt", "dphi_ttbar", "dphi_leptop_met", "dphi_hadtop_met",
#              #"m_jet1_jet2", "m_jet_jet_min", "m_jet_jet_max",
#              #"RJR_RISR", "RJR_MS", "RJR_PTISR", "RJR_nBJet",
#              #"stxe_trigger", "el_trigger", "mu_trigger", "lep_trig_req",
#              #"event_number", "run_number", "lumi_block", "mc_channel_number", "bcid",
#              #"tt_cat",
#              #"weight",
#              #"xs_weight",
#              #"sf_total",
#              #"weight_sherpa22_njets",
#              "Event",
#              "eventWeight"
#]

CHUNKSIZE = 100000

def filesize(num):
  for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
    if abs(num) < 1024.0:
      return "%3.1f%sB" % (num, unit)
    num /= 1024.0
  return "%.1fYiB" % num

def main():
  global CUT
  if not len(sys.argv) == 3:
    print "Usage: prepareSample.py <source directory> <destination directory>"
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
  inFiles = [os.path.relpath(os.path.join(d, f), src) for (d, _, files) in os.walk(src) for f in files if f.endswith(".root")] 

  print "Going to preprocess following files\nfrom\n\t{src}\nto\n\t{dest}\n(without overwriting existing files)\n\n- {files}" \
    .format(src=src, dest=dest, files="\n- ".join(inFiles))

  #while True:
  #  i = raw_input("Are you okay with that? (y|n) ").strip().lower()
  #  if i == "y":
  #    break
  #  elif i == "n":
  #    return

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
      if ("_ML" in name) or ("_lumi" in name):
        continue

      print "\nDEBUG: Processing " + name
      t = f.Get(name)

      nevents = t.GetEntries()
      idx = 0
      
      print 'This file has {} entries and chunks of size {} are used, resulting in {} loops.\n'.format(nevents, CHUNKSIZE, nevents//CHUNKSIZE + 1 )    
      
      for start in range(0, nevents, CHUNKSIZE):
        print 'Processing loop {} out of {}'.format(idx+1, nevents//CHUNKSIZE + 1)

        chunk = rn.tree2array(t, branches=VAR, start=start, stop=start+CHUNKSIZE, selection=CUT)
        df = pd.DataFrame(chunk)
  
        outFile = fDest.replace(".root","_"+str(idx)+".h5")

        print "\n\tFile check! {}".format(os.path.join(dest, outFile))
        print "\t(Number of events, Number of branches): ",df.shape

        store = pd.HDFStore(outFile, "w")
        store.put(name, df, data_columns=df.columns)
        store.close()

        idx = idx + 1

    f.Close()


  writeInfo = os.path.join(dest, "info.txt")
  #if os.path.exists(writeInfo):
    #while True:
    #  i = raw_input("'{}' exists. Should it be overwritten? (y|n) ".format(writeInfo)).strip().lower()
    #  if i == "y":
    #    break
    #  elif i == "n":
    #    return

  with open(writeInfo, "w") as f:
    f.write("Preprocess Info!\nCut: {}".format(CUT))
    f.write("\nVariables:")
    for var in VAR:
      f.write("\n"+var)

if __name__ == "__main__":
  main()

