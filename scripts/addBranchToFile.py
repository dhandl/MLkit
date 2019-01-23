#!/usr/bin/env python

import ROOT
import os
import sys
from variables import new_var
from array import array

def filesize(num):
  for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
    if abs(num) < 1024.0:
      return "%3.1f%sB" % (num, unit)
    num /= 1024.0
  return "%.1fYiB" % num

def main():
  print "Usage: addBranchToFile.py <source directory>"
  src = sys.argv[1]

  if not os.path.isdir(src):
    print "No such folder '{}'".format(src)
    return

  # going to check whether EventWeight will be applied
  for v in new_var:
    if v["name"] is "EventWeight":
      lumi = float(raw_input("Going to apply total event weight!\nPlease set int. luminosity(pb^-1): "))

  # get all .root files in all subdirectories of <src>
  inFiles = [os.path.relpath(os.path.join(d, f), src) for (d, _, files) in os.walk(src) for f in files if f.endswith("root")] 

  for f in inFiles:
    fSrc= os.path.join(src, f)

    f = ROOT.TFile(fSrc, "UPDATE")

    # Get all trees in this file
    for name in set([k.GetName() for k in f.GetListOfKeys() if k.GetClassName() == "TTree"]):
      print "\nDEBUG: Copying " + name
      t = f.Get(name)

      for v in new_var:
        v["branch"] = t.Branch(v["name"], v["array"], v["name"]+"/"+v["dataType"])

      n = t.GetEntries()
      for i in xrange(n):
        if (i%10000 == 0):
          print "At %i-th event of %i total events in sample %s!"%(i,n,name)
        t.GetEntry(i)
        assert (v.has_key('varString') or v.has_key('varFunc')), "Error: Did not specify 'varString' or 'varFunc' for variable %s" % repr(v)
        assert not (v.has_key('varString') and v.has_key('varFunc')), "Error: Specified both 'varString' and 'varFunc' for variable %s" % repr(v)
        for v in new_var:
          varValue = getVarValue(t, v['varString']) if v.has_key('varString') else v['varFunc'](t)
          if v["name"] is "EventWeight":
            varValue = lumi * varValue
          v["array"][0] = varValue
          v["branch"].Fill()

      t.Write()
      f.Write()
      f.Close()

    print "OK Saved {}".format(filesize(os.stat(fSrc).st_size))

  varInfo = os.path.join(src, "new_var.txt")
  if os.path.exists(varInfo):
    while True:
      i = raw_input("'{}' exists. Should it be overwritten? (y|n) ".format(varInfo)).strip().lower()
      if i == "y":
        break
      elif i == "n":
        return

  with open(varInfo, "w") as f:
    for v in new_var:
      f.write(v["name"])

if __name__ == "__main__":
  main()
