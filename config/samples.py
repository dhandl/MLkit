from collections import namedtuple

#input = '/gpfs/scratch/pr62re/di36jop/samples/ttbar_rew/TRUTH3/hdf5/'
#input = '/project/etp5/dhandl/samples/ttbar_rew/TRUTH3/hdf5/'
inputMC15 = '/project/etp5/dhandl/samples/SUSY/Stop1L/EarlyRun2/hdf5/cut_mt30_met60_preselection/'
inputMC16 = '/project/etp5/dhandl/samples/SUSY/Stop1L/FullRun2/hdf5/cut_mt30_met60_preselection_new/'

testinput = '/project/etp5/dhandl/samples/SUSY/Stop1L/FullRun2/hdf5/test/'

# define your samples here
SignalMC16 = [
  {'name':'stop_bWN_450_300_truth', 'path':inputMC16+'stop_bWN_450_300_truth/'}
  #{'name':'stop_bWN_450_300_mc16a', 'path':inputMC16+'stop_bWN_450_300_mc16a/'},
  #{'name':'stop_bWN_450_300_mc16d', 'path':inputMC16+'stop_bWN_450_300_mc16d/'}
  #{'name':'stop_bWN_450_300', 'path':inputMC16+'stop_bWN_450_300_TruthSmeared/'}
  #{'name':'stop_bWN_450_300_truth', 'path':testinput+'stop_bWN_450_300_truth/'}
]

Signal1 = [
  #{'name':'powheg_ttbar_TRUTH3', 'path':input+'powheg_ttbar_TRUTH3/'}
  {'name':'stop_bWN_250_100', 'path':inputMC15+'stop_bWN_250_100/'},
  {'name':'stop_bWN_300_150', 'path':inputMC15+'stop_bWN_300_150/'},
  {'name':'stop_bWN_350_200', 'path':inputMC15+'stop_bWN_350_200/'},
  {'name':'stop_bWN_400_250', 'path':inputMC15+'stop_bWN_400_250/'},
  {'name':'stop_bWN_450_300', 'path':inputMC15+'stop_bWN_450_300/'},
  {'name':'stop_bWN_500_350', 'path':inputMC15+'stop_bWN_500_350/'},
  {'name':'stop_bWN_550_400', 'path':inputMC15+'stop_bWN_550_400/'},
  {'name':'stop_bWN_600_450', 'path':inputMC15+'stop_bWN_600_450/'},
  {'name':'stop_bWN_650_500', 'path':inputMC15+'stop_bWN_650_500/'}
]


Signal2 = [
  #{'name':'powheg_ttbar_TRUTH3', 'path':input+'powheg_ttbar_TRUTH3/'}
  {'name':'stop_bWN_250_130', 'path':inputMC15+'stop_bWN_250_130/'},
  {'name':'stop_bWN_300_180', 'path':inputMC15+'stop_bWN_300_180/'},
  {'name':'stop_bWN_350_230', 'path':inputMC15+'stop_bWN_350_230/'},
  {'name':'stop_bWN_400_280', 'path':inputMC15+'stop_bWN_400_280/'},
  {'name':'stop_bWN_450_330', 'path':inputMC15+'stop_bWN_450_330/'},
  {'name':'stop_bWN_500_380', 'path':inputMC15+'stop_bWN_500_380/'},
  {'name':'stop_bWN_550_430', 'path':inputMC15+'stop_bWN_550_430/'},
  {'name':'stop_bWN_600_480', 'path':inputMC15+'stop_bWN_600_480/'},
  {'name':'stop_bWN_650_530', 'path':inputMC15+'stop_bWN_650_530/'}
]

Signal3 = [
  #{'name':'powheg_ttbar_TRUTH3', 'path':input+'powheg_ttbar_TRUTH3/'}
  {'name':'stop_bWN_250_160', 'path':inputMC15+'stop_bWN_250_160/'},
  {'name':'stop_bWN_300_210', 'path':inputMC15+'stop_bWN_300_210/'},
  {'name':'stop_bWN_350_260', 'path':inputMC15+'stop_bWN_350_260/'},
  {'name':'stop_bWN_400_310', 'path':inputMC15+'stop_bWN_400_310/'},
  {'name':'stop_bWN_450_360', 'path':inputMC15+'stop_bWN_450_360/'},
  {'name':'stop_bWN_550_460', 'path':inputMC15+'stop_bWN_550_460/'},
  {'name':'stop_bWN_600_510', 'path':inputMC15+'stop_bWN_600_510/'},
  {'name':'stop_bWN_650_560', 'path':inputMC15+'stop_bWN_650_560/'}
]

Signal4 = [
  #{'name':'powheg_ttbar_TRUTH3', 'path':input+'powheg_ttbar_TRUTH3/'}
  {'name':'stop_bWN_350_185', 'path':inputMC15+'stop_bWN_350_185/'},
  {'name':'stop_bWN_400_235', 'path':inputMC15+'stop_bWN_400_235/'},
  {'name':'stop_bWN_450_285', 'path':inputMC15+'stop_bWN_450_285/'},
  {'name':'stop_bWN_500_335', 'path':inputMC15+'stop_bWN_500_335/'},
  {'name':'stop_bWN_550_385', 'path':inputMC15+'stop_bWN_550_385/'},
  {'name':'stop_bWN_600_435', 'path':inputMC15+'stop_bWN_600_435/'},
  {'name':'stop_bWN_650_485', 'path':inputMC15+'stop_bWN_650_485/'}
]


#Signal1.extend(Signal2)
#Signal1.extend(Signal3)

Signal = SignalMC16

Signal_Evaluate = Signal1
Signal_Evaluate.extend(Signal2)
Signal_Evaluate.extend(Signal3)
Signal_Evaluate.extend(Signal4)

Background = [
  #{'name':'ttbar_radHi_TRUTH3',  'path':input+'TestBkg/'}
  #{'name':'ttbar_radLo_TRUTH3', 'path':input+'ttbar_radLo_TRUTH3/'}
  {'name':'mc16a_ttbar', 'path':inputMC16+'mc16a_ttbar/'},
  {'name':'mc16d_ttbar', 'path':inputMC16+'mc16d_ttbar/'},
  {'name':'mc16a_singletop', 'path':inputMC16+'mc16a_singletop/'},
  {'name':'mc16d_singletop', 'path':inputMC16+'mc16d_singletop/'},
  {'name':'mc16a_wjets', 'path':inputMC16+'mc16a_wjets/'},
  {'name':'mc16d_wjets', 'path':inputMC16+'mc16d_wjets/'}
  #{'name':'mc16a_ttbar', 'path':testinput+'mc16a_ttbar/'},
]
