import numpy as np

def getSumW2(x, w, binning):
  weight = []
  
  #Sanity checks
  if x.shape[0] != w.shape[0]:
    print 'Variable array and weight array have different size'
    return 0
  if not type(binning) == list:
    print 'Binning should be a list of type [nbins, start, stop]'
    return 0

  db = (binning[2] - binning[1]) / binning[0]
  bins = np.arange(binning[1], binning[2]+db, db)
  b_indices = np.digitize(x, bins[1:])

  for i in range(len(bins[1:])):
    ws = w[np.where(b_indices==i)[0]]
    error = np.sqrt(np.sum(ws**2.))
    weight.append(error)

  return np.array(weight)
