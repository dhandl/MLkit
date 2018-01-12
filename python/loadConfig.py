from collections import namedtuple
from LoadData import *

Algorithm = namedtuple("Algorithm", "name modelname options")

# save model in directory
saveDir = "../TrainedModels/models/"
fileSuffix = "09012018_firstTry_multiclassTRUE_"
# load data from path 
etpDir = "/project/etp5/dhandl/samples/SUSY/Stop1L/"
cpapDir = "/gpfs/scratch/pr62re/di36jop/samples/ttbar_rew/TRUTH3/"

preselection = "(n_lep==1) && (lep_pt[0]>=25e3) && (n_jet>=4) && (jet_pt[0]>25e3) && (jet_pt[1]>25e3) && (jet_pt[2]>25e3) && (jet_pt[3]>25e3)"

# define your input variables and the weights
nvar = [
        'met',
        'n_jet',
        'jet_pt[0]',
        'jet_pt[1]',
        'jet_pt[2]',
        'jet_pt[3]',
        'dr_lep_jet_min',
        'dr_lep_jet_max',
        'dr_jet_jet_min',
        'dr_jet_jet_max',
        'm_jet1_jet2',
        'm_jet_jet_min',
        'm_jet_jet_max'
]

weights = [
            'weight',
            'xs_weight'
]

lumi = 36100.

# define your input samples here
Signal = [
  {'name':'powheg_ttbar_TRUTH3', 'tree':'powheg_ttbar_TRUTH3_Nom', 'path':cpapDir, 'cut':preselection, 'branches':nvar, 'weights':weights, 'lumi':lumi},
]

Background = [
  {'name':'ttbar_radHi_TRUTH3', 'tree':'ttbar_radHi_TRUTH3_Nom', 'path':cpapDir, 'cut':preselection, 'branches':nvar, 'weights':weights, 'lumi':lumi},
  {'name':'ttbar_radLo_TRUTH3', 'tree':'ttbar_radLo_TRUTH3_Nom', 'path':cpapDir, 'cut':preselection, 'branches':nvar, 'weights':weights, 'lumi':lumi},
]

# define your ML algorithm here

analysis = [
            Algorithm('BDT',
                      fileSuffix+'AdaBoost_d2_mspl10_nEst50_lr0p01',
                      {
                       'classifier':'AdaBoost',
                       'max_depth':2,
                       'min_samples_leaf':10,
                       'n_estimators':50,
                       'learning_rate':0.01
                      }
            ),
            Algorithm('NN',
                      fileSuffix+'DNN_ADAM_layer16-16_epochs50_batch512',
                      {
                      'layers':[16,16,1],
                      'ncycles':50,
                      'batchSize':512,
                      'dropout':0.3,
                      'optimizer':'adam',
                      'activation':'relu',
                      'initializer':'glorot_uniform',
                      'learningRate':0.01,
                      'decay':0.0,
                      'momentum':0.0,
                      'nesterov':False,
                      'multiclassification':False
                      }
            ),
            #Algorithm('RNN',
            #          {}
            #)
]

