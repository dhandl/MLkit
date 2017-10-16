#!/usr/bin/env python

import ROOT
import os, copy, sys
import datetime
import pandas as pd
from variables import variables
from root2pandas import root2pandas

now = datetime.datetime.now()

def filesize(num):
  for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
    if abs(num) < 1024.0:
      return "%3.1f%sB" % (num, unit)
    num /= 1024.0
  return "%.1fYiB" % num

def main():
  print "Usage: root2hdf.py <source directory> <destination directory>"
  src, dest = sys.argv[1:]

  if not os.path.isdir(src):
    print "No such folder '{}'".format(src)
    return

  if not os.path.isdir(dest):
    print "No such folder '{}'".format(dest)
    return

  # get all .root files in all subdirectories of <src>
  inFiles = [os.path.relpath(os.path.join(d, f), src) for (d, _, files) in os.walk(src) for f in files if f.endswith("root")] 

  print "Going to convert following files\nfrom\n\t{src}\nto\n\t{dest}\n(without overwriting existing files)\n to .h5 data format".format(src=src, dest=dest)

  while True:
    i = raw_input("Are you okay with that? (y|n) ").strip().lower()
    if i == "y":
      break
    elif i == "n":
      return

  for f in inFiles:
    fSrc, fDest = os.path.join(src, f), os.path.join(dest, f)

    if not os.path.exists(os.path.dirname(fDest)):
      os.makedirs(os.path.dirname(fDest))

    if os.path.exists(fDest):
      print "Skipping " + f
      continue

    print "Converting '{}'...".format(f),

    rootfile = ROOT.TFile(fSrc)

    # Get all trees in this file
    for name in set([k.GetName() for k in rootfile.GetListOfKeys() if k.GetClassName() == "TTree"]):
      print "\nDEBUG: Converting " + name

      # Create data frame from .root file
      outFile = f.replace(".root", ".h5")
      #hdf_file = pd.HDFStore(outFile)
      df = root2pandas(fSrc, name)
      #hdf_file.put(name, df)

      # save a pandas df to hdf5 (better to first convert it back to ndarray, to be fair)
      import deepdish.io as io
      io.save(os.path.join(dest, outFile), df)

      # let's load it back in to make sure it actually worked!
      new_df = io.load(os.path.join(dest, outFile))
      # -- check the shape again -- nice check to run every time you create a df
      print "File check!"
      print "(Number of events, Number of branches): ",new_df.shape

if __name__ == "__main__":
  main()
