from collections import namedtuple

#input = '/gpfs/scratch/pr62re/di36jop/samples/ttbar_rew/TRUTH3/hdf5/'
#input = '/project/etp5/dhandl/samples/ttbar_rew/TRUTH3/hdf5/'
input = '/project/etp5/dhandl/samples/SUSY/Stop1L/hdf5/cut_mt30_met60_preselection/'

# define your samples here
Signal1 = [
  #{'name':'powheg_ttbar_TRUTH3', 'path':input+'powheg_ttbar_TRUTH3/'}
  {'name':'stop_bWN_250_100', 'path':input+'stop_bWN_250_100/'},
  {'name':'stop_bWN_300_150', 'path':input+'stop_bWN_300_150/'},
  {'name':'stop_bWN_350_200', 'path':input+'stop_bWN_350_200/'},
  {'name':'stop_bWN_400_250', 'path':input+'stop_bWN_400_250/'},
  {'name':'stop_bWN_450_300', 'path':input+'stop_bWN_450_300/'},
  {'name':'stop_bWN_500_350', 'path':input+'stop_bWN_500_350/'},
  {'name':'stop_bWN_550_400', 'path':input+'stop_bWN_550_400/'},
  {'name':'stop_bWN_600_450', 'path':input+'stop_bWN_600_450/'},
  {'name':'stop_bWN_650_500', 'path':input+'stop_bWN_650_500/'}
]


Signal2 = [
  #{'name':'powheg_ttbar_TRUTH3', 'path':input+'powheg_ttbar_TRUTH3/'}
  {'name':'stop_bWN_250_130', 'path':input+'stop_bWN_250_130/'},
  {'name':'stop_bWN_300_180', 'path':input+'stop_bWN_300_180/'},
  {'name':'stop_bWN_350_230', 'path':input+'stop_bWN_350_230/'},
  {'name':'stop_bWN_400_280', 'path':input+'stop_bWN_400_280/'},
  {'name':'stop_bWN_450_330', 'path':input+'stop_bWN_450_330/'},
  {'name':'stop_bWN_500_380', 'path':input+'stop_bWN_500_380/'},
  {'name':'stop_bWN_550_430', 'path':input+'stop_bWN_550_430/'},
  {'name':'stop_bWN_600_480', 'path':input+'stop_bWN_600_480/'},
  {'name':'stop_bWN_650_530', 'path':input+'stop_bWN_650_530/'}
]

Signal3 = [
  #{'name':'powheg_ttbar_TRUTH3', 'path':input+'powheg_ttbar_TRUTH3/'}
  {'name':'stop_bWN_250_160', 'path':input+'stop_bWN_250_160/'},
  {'name':'stop_bWN_300_210', 'path':input+'stop_bWN_300_210/'},
  {'name':'stop_bWN_350_260', 'path':input+'stop_bWN_350_260/'},
  {'name':'stop_bWN_400_310', 'path':input+'stop_bWN_400_310/'},
  {'name':'stop_bWN_450_360', 'path':input+'stop_bWN_450_360/'},
  {'name':'stop_bWN_550_460', 'path':input+'stop_bWN_550_460/'},
  {'name':'stop_bWN_600_510', 'path':input+'stop_bWN_600_510/'},
  {'name':'stop_bWN_650_560', 'path':input+'stop_bWN_650_560/'}
]


#Signal1.extend(Signal2)
#Signal1.extend(Signal3)

Signal = Signal1

Signal_Evaluate = Signal1
Signal_Evaluate.extend(Signal2)
Signal_Evaluate.extend(Signal3)

Background = [
  #{'name':'ttbar_radHi_TRUTH3',  'path':input+'TestBkg/'}
  #{'name':'ttbar_radLo_TRUTH3', 'path':input+'ttbar_radLo_TRUTH3/'}
  {'name':'powheg_ttbar', 'path':input+'powheg_ttbar/'},
  {'name':'powheg_singletop', 'path':input+'powheg_singletop/'},
  {'name':'sherpa22_Wjets', 'path':input+'sherpa22_Wjets/'}
]
