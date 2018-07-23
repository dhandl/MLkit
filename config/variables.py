
preselection = [ 
                {'name':'n_jet',  'threshold':4,      'type':'geq'},
                {'name':'n_bjet',  'threshold':1,      'type':'geq'},
                {'name':'met',    'threshold':100e3,  'type':'geq'},
                {'name':'mt',    'threshold':90e3,  'type':'geq'},
                {'name':'n_lep',  'threshold':1,      'type':'exact'}
                #{'name':'dphi_jet0_ptmiss', 'threshold':1.8,    'type':'geq'},
                #{'name':'dphi_jet1_ptmiss', 'threshold':0.4,    'type':'greater'}
                #{'name':'ht',   'threshold':200e3,  'type':'geq'},
                #{'name':'dphi_b_lep_max',   'threshold':2.8,  'type':'leq'}
                #{'name':'dphi_b_lep_max',   'threshold':2.6,  'type':'condition',   'variable':'ht',   'lessthan':225e3,   'morethan': 275e3}
                #{'name':'dr_bjet_lep',    'threshold':2.5,  'type':'leq'}
                #{'name':'dphi_met_lep',    'threshold':1.8,  'type':'geq'}
                #{'name':'tt_cat_TRUTH3',  'threshold':0,      'type':'geq'},
                #{'name':'tt_cat_TRUTH3',  'threshold':6,      'type':'leq'},
                #{'name':'tt_cat_TRUTH3',  'threshold':1,      'type':'not'},
                #{'name':'tt_cat_TRUTH3',  'threshold':4,      'type':'not'}
]

preselection_evaluate = [ 
                {'name':'n_jet',  'threshold':4,      'type':'geq'},
                {'name':'n_bjet',  'threshold':1,      'type':'geq'},
                {'name':'met',    'threshold':250e3,  'type':'geq'},
                {'name':'mt',    'threshold':90e3,  'type':'geq'},
                {'name':'n_lep',  'threshold':1,      'type':'exact'}
                #{'name':'dphi_jet0_ptmiss', 'threshold':0.4,    'type':'greater'},
                #{'name':'dphi_jet1_ptmiss', 'threshold':0.4,    'type':'greater'}
                #{'name':'ht',   'threshold':200e3,  'type':'geq'},
                #{'name':'dphi_b_lep_max',   'threshold':2.8,  'type':'leq'}
                #{'name':'dphi_b_lep_max',   'threshold':2.6,  'type':'condition',   'variable':'ht',   'lessthan':225e3,   'morethan': 275e3}
                #{'name':'dr_bjet_lep',    'threshold':2.5,  'type':'leq'}
                #{'name':'dphi_met_lep',    'threshold':2,  'type':'leq'}
                #{'name':'tt_cat_TRUTH3',  'threshold':0,      'type':'geq'},
                #{'name':'tt_cat_TRUTH3',  'threshold':6,      'type':'leq'},
                #{'name':'tt_cat_TRUTH3',  'threshold':1,      'type':'not'},
                #{'name':'tt_cat_TRUTH3',  'threshold':4,      'type':'not'}
]

lumi = 140e3

#High level variables

nvar = [
        'ht',
        #'jet_pt[0]',
        'bjet_pt[0]',
        'amt2',
        'mt',
        'met',
        'dphi_met_lep',
        'dphi_b_lep_max',
        'dphi_jet0_ptmiss',
        #'met_sig',
        #'met_proj_lep',
        #'ht_sig',
        'm_bl',
        'dr_bjet_lep'
        #'mT_blMET' #15vars
        #'n_jet',
        #'n_bjet'
        #'jet_pt[1]',
        #'jet_pt[2]',
        #'jet_pt[3]',
        #'lep_pt[0]',
        #'dr_jet_jet_max',
        #'ttbar_pt',
        #'ttbar_dphi',
        #'m_jet1_jet2',
        #'m_jet_jet_min',
        #'m_jet_jet_max'
        #'tt_cat_TRUTH3'
        #'n_lep',
        #'lep_eta[0]',
        #'lep_phi[0]',
        #'jet_eta[0]',
        #'jet_phi[0]',
        #'dphi_jet1_ptmiss',
        #'dphi_jet2_ptmiss',
        #'dphi_jet3_ptmiss',
        #'dphi_min_ptmiss',
        #'dphi_b_ptmiss_max'
        ]

#Low level variables

#nvar = [
        #'met',
        #'met_phi',
        #'n_jet',
        #'n_bjet',
        #'jet_pt[0]',
        #'jet_eta[0]',
        #'jet_phi[0]',
        #'jet_e[0]',
        #'jet_mv2c10[0]',
        #'jet_pt[1]',
        #'jet_eta[1]',
        #'jet_phi[1]',
        #'jet_e[1]',
        #'jet_mv2c10[1]',
        #'jet_pt[2]',
        #'jet_eta[2]',
        #'jet_phi[2]',
        #'jet_e[2]',
        #'jet_mv2c10[2]',
        #'jet_pt[3]',
        #'jet_eta[3]',
        #'jet_phi[3]',
        #'jet_e[3]',
        #'jet_mv2c10[3]',
        #'lep_pt[0]',
        #'lep_eta[0]',
        #'lep_phi[0]',
        #'lep_e[0]'
#]

weight = [
          'weight',
          'xs_weight',
          'sf_total',
          'weight_sherpa22_njets'
]
