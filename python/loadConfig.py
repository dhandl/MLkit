from collections import namedtuple
from LoadData import *

Algorithm = namedtuple("Algorithm", "name modelname options")

# save model in directory
saveDir = "../TrainedModels/models/"
fileSuffix = "18102017_highMethighMtPreselection_multiclassFALSE_"
# load data from path 
loadDir = "/project/etp5/dhandl/samples/SUSY/Stop1L/"

preselection = "(dphi_jet0_ptmiss > 0.4) && (dphi_jet1_ptmiss > 0.4) && !((mT2tauLooseTau_GeV > -0.5) && (mT2tauLooseTau_GeV < 80)) && (n_jet>=4) && (n_bjet>=1) && (jet_pt[0]>25e3) && (jet_pt[1]>25e3) && (jet_pt[2]>25e3) && (jet_pt[3]>25e3) && (mt>60e3) && (met>120e3)"

# define your input variables and the weights
nvar = [
        'met',
        'mt',
        'dphi_met_lep',
        'amt2',
        'n_jet',
        'n_bjet',
        'ht_sig',
        'jet_pt[0]',
        'jet_pt[1]',
        'jet_pt[2]',
        'jet_pt[3]',
        'bjet_pt[0]',
        'm_bl',
        #'mT250',
        #'dphi_b_lep_max',
        #'dphi_b_ptmiss_max',
        #'dr_bjet_lep',
        #'mT_b1lMET',
        #'m_top_chi2',
        #'met_proj_lep',
        'met_sig',
        #'mt2stop'
]

weights = [
            'weight',
            'xs_weight',
            'sf_total',
            'weight_sherpa22_njets'
]

lumi = 36100.

# define your input samples here
Signal = [
  {'name':'stop_bWN_350_185', 'tree':'stop_bWN_350_185_Nom', 'path':loadDir, 'cut':preselection, 'branches':nvar, 'weights':weights, 'lumi':lumi},
  {'name':'stop_bWN_350_200', 'tree':'stop_bWN_350_200_Nom', 'path':loadDir, 'cut':preselection, 'branches':nvar, 'weights':weights, 'lumi':lumi},
  {'name':'stop_bWN_350_230', 'tree':'stop_bWN_350_230_Nom', 'path':loadDir, 'cut':preselection, 'branches':nvar, 'weights':weights, 'lumi':lumi},
  {'name':'stop_bWN_350_260', 'tree':'stop_bWN_350_260_Nom', 'path':loadDir, 'cut':preselection, 'branches':nvar, 'weights':weights, 'lumi':lumi},
  {'name':'stop_bWN_400_235', 'tree':'stop_bWN_400_235_Nom', 'path':loadDir, 'cut':preselection, 'branches':nvar, 'weights':weights, 'lumi':lumi},
  {'name':'stop_bWN_400_250', 'tree':'stop_bWN_400_250_Nom', 'path':loadDir, 'cut':preselection, 'branches':nvar, 'weights':weights, 'lumi':lumi},
  {'name':'stop_bWN_400_280', 'tree':'stop_bWN_400_280_Nom', 'path':loadDir, 'cut':preselection, 'branches':nvar, 'weights':weights, 'lumi':lumi},
  {'name':'stop_bWN_400_310', 'tree':'stop_bWN_400_310_Nom', 'path':loadDir, 'cut':preselection, 'branches':nvar, 'weights':weights, 'lumi':lumi}
]

Background = [
  {'name':'powheg_ttbar', 'tree':'powheg_ttbar_Nom', 'path':loadDir, 'cut':preselection, 'branches':nvar, 'weights':weights, 'lumi':lumi}
  #{'name':'sherpa22_Wjets', 'tree':'sherpa22_Wjets_Nom', 'path':loadDir, 'cut':preselection, 'branches':nvar, 'weights':weights, 'lumi':lumi}
]

# define your ML algorithm here

analysis = [
            #Algorithm('BDT',
            #          fileSuffix+'AdaBoost_d2_mspl10_nEst50_lr0p01',
            #          {
            #           'classifier':'AdaBoost',
            #           'max_depth':2,
            #           'min_samples_leaf':10,
            #           'n_estimators':50,
            #           'learning_rate':0.01
            #          }
            #),
            Algorithm('NN',
                      fileSuffix+'DNN_ADAM_layer40_epochs75_batch512_lr0p01',
                      {
                      'layers':[40,1],
                      'ncycles':75,
                      'batchSize':512,
                      'dropout':0.3,
                      'optimizer':'adam',
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

