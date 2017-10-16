#!/usr/bin/env python
import ROOT
import os
import sys

CUT = "(dphi_jet0_ptmiss > 0.4) && (dphi_jet1_ptmiss > 0.4) && (n_jet>=4) && (n_bjet>0) && (jet_pt[0]>50e3) && (jet_pt[1]>25e3) && (jet_pt[2]>25e3) && (jet_pt[3]>25e3) && (mt>30e3) && !((mT2tauLooseTau_GeV > -0.5) && (mT2tauLooseTau_GeV < 80)) && (met >120e3)"


def filesize(num):
	for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
		if abs(num) < 1024.0:
			return "%3.1f%sB" % (num, unit)
		num /= 1024.0
	return "%.1fYiB" % num

def main():
	global CUT
	if not len(sys.argv) == 3:
		print "Usage: cut-tree.py <source directory> <destination directory> (<cut>)"
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

	print "Going to copy following files\nfrom\n\t{src}\nto\n\t{dest}\n(without overwriting existing files)\nand applying the following cut:\n\n{cut}\n\n- {files}" \
		.format(src=src, dest=dest, files="\n- ".join(inFiles), cut=CUT)

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

		print "Copying '{}'...".format(f),

		f = ROOT.TFile(fSrc)

		# Get all trees in this file
		for name in set([k.GetName() for k in f.GetListOfKeys() if k.GetClassName() == "TTree"]):
			print "\nDEBUG: Copying " + name
			t = f.Get(name)

			# Open destination file for this tree. This is important as otherwise the tree would get written to
			# memory by default when doing CopyTree
			fCopy = ROOT.TFile(fDest, "RECREATE")
			fCopy.cd()

			tCopy = t.CopyTree(CUT)
			tCopy.AutoSave()

		print "OK Saved {}".format(filesize(os.stat(fSrc).st_size - os.stat(fDest).st_size))

	cutInfo = os.path.join(dest, "cut.txt")
	if os.path.exists(cutInfo):
		while True:
			i = raw_input("'{}' exists. Should it be overwritten? (y|n) ".format(cutInfo)).strip().lower()
			if i == "y":
				break
			elif i == "n":
				return

	with open(cutInfo, "w") as f:
		f.write(CUT)


if __name__ == "__main__":
	main()

