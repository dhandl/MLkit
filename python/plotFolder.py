import os
import startPlot as sp

def plotFolder():
    failures = []
    foldercontent = os.listdir('TrainedModels/models')
    models = [x for x in foldercontent if x[-3:]=='.h5']
    for modelDir in models:
        try:
            sp.startPlot('TrainedModels/models/'+modelDir,save=True)
        except (IOError, IndexError):
            failures.append(modelDir)
    print('Failures: ', failures)