import os
import startPlot as sp
import timer

def plotFolder():
    t = timer.Timer()
    t.start()
    
    failures = []
    foldercontent = os.listdir('TrainedModels/models')
    models = [x for x in foldercontent if x[-3:]=='.h5']
    for modelDir in models:
        try:
            sp.startPlot('TrainedModels/models/'+modelDir,save=True)
        except (IOError, IndexError):
            failures.append(modelDir)
    print '---------Failures---------'
    print(failures)
    
    # end timer and print time
    t.stop()
    t0 = t.elapsed
    t.reset()
    runtimeSummary(t0)
    
def runtimeSummary(t0):
  hour = t0 // 3600
  t0 %= 3600
  minutes = t0 // 60
  t0 %= 60
  seconds = t0

  print '-----Runtime Summary -----'
  print 'Job ran %d h:%d min:%d sec' % ( hour, minutes, seconds)
  print '--------------------------'