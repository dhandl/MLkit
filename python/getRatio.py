import numpy as np
import matplotlib.pyplot as plt

def getRatio(s_hist, s_bins, s_w, b_hist, b_bins, b_w, color):
  S = s_hist.copy()
  B = b_hist.copy()

  # Sanity checks
  if len(s_bins) != len(b_bins) or len(b_bins) <=0:
    print "Signal and Bkg samples with different number of bins: S(" + str(len(s_bins))+ ") B(" + str(len(b_bins)) + ")"
    return 0

  if s_bins.max() != b_bins.max() or s_bins.min() != b_bins.min() or s_bins.max() <= s_bins.min():
    print "Edges of histos are not right: Smin(" + str(s_bins.min()) + ")  Bmin(" + str(b_bins.min()) + ")"
    print "Smax(" + str(s_bins.max()) + ")  Bmax(" + str(b_bins.max()) + ")"
    return 0
  
  ratio = np.ma.masked_invalid(S/B)
  err = np.ma.masked_invalid(ratio * np.sqrt((s_w/S)**2 + (b_w/B)**2))
  width = (s_bins[1] - s_bins[0])
  center = (s_bins[:-1] + s_bins[1:]) / 2
  plt.errorbar(center, ratio, yerr=err, fmt='o', color=color)

  return ratio
