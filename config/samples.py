from collections import namedtuple

#input = '/gpfs/scratch/pr62re/di36jop/samples/ttbar_rew/TRUTH3/hdf5/'
#input = '/project/etp5/dhandl/samples/ttbar_rew/TRUTH3/hdf5/'
input = '/project/etp5/dhandl/samples/SUSY/Stop1L/hdf5/cut_mt30_met60_preselection/'

# define your samples here
Signal = [
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

Background = [
  #{'name':'ttbar_radHi_TRUTH3',  'path':input+'TestBkg/'}
  #{'name':'ttbar_radLo_TRUTH3', 'path':input+'ttbar_radLo_TRUTH3/'}
  {'name':'powheg_ttbar', 'path':input+'powheg_ttbar/'},
  {'name':'powheg_singletop', 'path':input+'powheg_singletop/'},
  {'name':'sherpa22_Wjets', 'path':input+'sherpa22_Wjets/'}
]
