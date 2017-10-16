#!/usr/bin/env python

import ROOT
import os, copy, sys
import datetime
from variables import variables

now = datetime.datetime.now()

##_orig_argv = sys.argv[:]
##sys.argv = [_orig_argv[0]]
##
##sys.argv = _orig_argv
##
##def parse_options():
##  import argparse
##
##  parser = argparse.ArgumentParser()
##  parser.add_argument("-f", "--inputFile", help="The input file that needs to be slimmed")
##  parser.add_argument("-t", "--treeName", help="The name of the tree that needs to be slimmed")
##  parser.add_argument("-o", "--outputFile", help="The name of the slimmed output file")
##
##  opts = parser.parse_args()
##
##  if not os.path.exists(opts.inputFile):
##    print "File %s not found! "%(opts.inputFile)
##    sys.exit(1)
##
##  return opts
##
##def main():
##  opts = parse_options()
##
##  inFile = ROOT.TFile(opts.inputFile)
##  tree = inFile.Get(opts.treeName)
##  
##  tree.SetBranchStatus("*", 0)
##  for var in variables:
##    tree.SetBranchStatus(var, 1)
##
##  newFile = ROOT.TFile(opts.outputFile, "RECREATE")
##  newTree = tree.CloneTree()
##
##  newFile.Write()  
##
##if __name__ == '__main__':
##  main()

##########################################################################################################


def filesize(num):
  for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
    if abs(num) < 1024.0:
      return "%3.1f%sB" % (num, unit)
    num /= 1024.0
  return "%.1fYiB" % num

def main():
  print "Usage: slimRootFile.py <source directory> <destination directory>"
  src, dest = sys.argv[1:]

  if not os.path.isdir(src):
    print "No such folder '{}'".format(src)
    return

  if not os.path.isdir(dest):
    print "No such folder '{}'".format(dest)
    return

  # get all .root files in all subdirectories of <src>
  inFiles = [os.path.relpath(os.path.join(d, f), src) for (d, _, files) in os.walk(src) for f in files if f.endswith("root")] 

  print "Going to slim following files\nfrom\n\t{src}\nto\n\t{dest}\n(without overwriting existing files)\nand applying all variables from python/variables.py" \
    .format(src=src, dest=dest)

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

    print "Slimming '{}'...".format(f),
##  inFile = ROOT.TFile(opts.inputFile)
##  tree = inFile.Get(opts.treeName)
##  
##  tree.SetBranchStatus("*", 0)
##  for var in variables:
##    tree.SetBranchStatus(var, 1)
##
##  newFile = ROOT.TFile(opts.outputFile, "RECREATE")
##  newTree = tree.CloneTree()
##
##  newFile.Write()  

    f = ROOT.TFile(fSrc)

    # Get all trees in this file
    for name in set([k.GetName() for k in f.GetListOfKeys() if k.GetClassName() == "TTree"]):
      print "\nDEBUG: Copying " + name
      t = f.Get(name)

      # Resetting all branches and only set the variables from variables.py
      t.SetBranchStatus("*", 0)
      for var in variables:
        t.SetBranchStatus(var, 1)

      # Open destination file for this tree. This is important as otherwise the tree would get written to
      # memory by default when doing CopyTree
      fCopy = ROOT.TFile(fDest, "RECREATE")
      fCopy.cd()

      tCopy = t.CloneTree()
      fCopy.Write()

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
    f.write("Slimmed root file on the '{}'\n".format(now.strftime("%Y-%m-%d %H:%M")))
    f.write("Variables applied:\n")
    for v in variables:
      f.write(v+"\n")


if __name__ == "__main__":
  main()
