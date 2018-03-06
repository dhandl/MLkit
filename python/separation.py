import numpy as np

def getSeparation(s_hist, s_bins, b_hist, b_bins):
  sep = 0.

  # Normalize objects
  S = s_hist*1./(s_hist.sum())
  B = b_hist*1./(b_hist.sum())

  # Sanity checks
  if len(s_bins) != len(b_bins) or len(b_bins) <=0:
    print "Signal and Bkg samples with different number of bins: S(" + str(len(s_bins))+ ") B(" + str(len(b_bins)) + ")"
    return 0

  if s_bins.max() != b_bins.max() or s_bins.min() != b_bins.min() or s_bins.max() <= s_bins.min():
    print "Edges of histos are not right: Smin(" + str(s_bins.min()) + ")  Bmin(" + str(b_bins.min()) + ")"
    print "Smax(" + str(s_bins.max()) + ")  Bmax(" + str(b_bins.max()) + ")"
    return 0

  nstep = s_bins.max()
  intBin = (s_bins.max() - s_bins.min())/nstep
  nS = S.sum()*intBin
  nB = B.sum()*intBin
  
  if nS > 0 and nB > 0 :
    for bins in range(0,int(nstep)):
      s = S[bins]/nS
      b = B[bins]/nB
      if s+b>0 : sep +=  0.5*(s - b)*(s - b)/(s + b)
      pass
    sep *= intBin
  else : print "histos with 0 entries: Snb(" + str(nS) + ") Bnb("+ str(nB) + ")"; sep = 0
  return sep
