#!/usr/bin/env python

import ROOT
import os
import sys
from array import array

TOTAL_LUMI = (3219.56 + 32988.1 + 44307.4 + 58450.1)
LUMI_FRACTION = [(3219.56 + 32988.1)/TOTAL_LUMI, (44307.4)/TOTAL_LUMI, (58450.1)/TOTAL_LUMI]
LUMI_BRANCH_NAME = "lumi_weight_new"
LUMI_ARRAY = array("f", [0])

def filesize(num):
  for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
    if abs(num) < 1024.0:
      return "%3.1f%sB" % (num, unit)
    num /= 1024.0
  return "%.1fYiB" % num

def main():
  global TOTAL_LUMI
  global LUMI_ARRAY
  global LUMI_BRANCH_NAME

  if len(sys.argv)>2:
    print "Usage: applySubLumi.py <source directory>"
    return

  src = sys.argv[1]

  if not os.path.isdir(src):
    print "No such folder '{}'".format(src)
    return

  # get all .root files in all subdirectories of <src>
  inFiles = [os.path.relpath(os.path.join(d, f), src) for (d, _, files) in os.walk(src) for f in files if f.endswith("root")] 

  for f in inFiles:
    print "\nApplying lumi weight on file: {}".format(os.path.join(src,f))

    fSrc= os.path.join(src, f)

    f = ROOT.TFile(fSrc, "UPDATE")

    # Get all trees in this file
    for name in set([k.GetName() for k in f.GetListOfKeys() if k.GetClassName() == "TTree"]):
      if ("data" in name):
        print "Skipping {} ...".format(name)
        continue

      t = f.Get(name)
      original_tree = name
      #friendName = t.GetName()+"_lumi"

      branches = [b.GetName() for b in t.GetListOfBranches()]
      inList = filter(lambda x: x == LUMI_BRANCH_NAME, branches)
      if len(inList) > 0:
        print "Skipping {} ...".format(name)
        continue
      
      sub_campaign = fSrc[fSrc.find("mc16") + 4]
      if sub_campaign == "a":
        lumi_weight = LUMI_FRACTION[0]
      elif sub_campaign == "d":
        lumi_weight = LUMI_FRACTION[1]
      elif sub_campaign == "e":
        lumi_weight = LUMI_FRACTION[2]
      
      print "\tFound sub-campaign: {}\n\tWill apply lumi weight: {}".format("mc16"+sub_campaign, lumi_weight)
    
      lumi_branch = t.Branch(LUMI_BRANCH_NAME, LUMI_ARRAY, LUMI_BRANCH_NAME+"/F")

      n = t.GetEntries()
      for i in xrange(n):
        if (i%100000 == 0):
          print "At %i-th event of %i total events in sample %s!"%(i,n,name)
        LUMI_ARRAY[0] = lumi_weight
        lumi_branch.Fill()

      #friendTree.Write("", ROOT.TObject.kOverwrite)
      #t.AddFriend(friendName)
      t.Write(name, ROOT.TObject.kOverwrite)
    #f.Write()
    f.Close()

if __name__ == "__main__":
  main()
