#!/usr/bin/env python

import ROOT
import os, copy, sys
from root2pandas import root2pandas

_orig_argv = sys.argv[:]
sys.argv = [_orig_argv[0]]

sys.argv = _orig_argv

def parse_options():
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--inputFile", help="The input .root file that needs to be converted")
  parser.add_argument("-t", "--treeName", help="The name of the tree in the input file")

  opts = parser.parse_args()

  if not os.path.exists(opts.inputFile):
    print "File %s not found! "%(opts.inputFile)
    sys.exit(1)

  return opts

def main():
  opts = parse_options()

  inFile = opts.inputFile
  tree = opts.treeName

  df = root2pandas(inFile, tree)
  # -- save a pandas df to hdf5 (better to first convert it back to ndarray, to be fair)
  
  import deepdish.io as io
  outFile = inFile.replace(".root", ".h5")
  io.save(outFile, df)

  # -- let's load it back in to make sure it actually worked!
  new_df = io.load(outFile)

  # -- check the shape again -- nice check to run every time you create a df
  print "File check!"
  print "(Number of events, Number of branches): ",new_df.shape

if __name__ == '__main__':
  main()
