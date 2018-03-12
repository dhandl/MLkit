import os

import numpy as np

def getReweight(df, prob, weights, bins):

  # Sanity checks
  if df.shape[0] != prob.shape[0]:
    print "Sample and corresponding probabilities have different size"
    return 0
  if len(weights) != len(bins[1:]):
    print "Weigth factors and weight bins have different size"
    return 0

  reweight = []
  low_edge = bins[:-1]
  high_edge = bins[1:]
  for i in range(len(prob)):
    if i % 10000 == 0:
      print "Reweighting event %i"%i

    for b in range(len(low_edge)):
      if (prob[i] > low_edge[b]) and (prob[i] <= high_edge[b]):
        w = weights[b]
        break 
      else:
        w = 1.

    reweight.append(w)

  return np.array(reweight)
