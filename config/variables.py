
preselection = [ 
                {'name':'n_jet',  'threshold':4,      'type':'geq'},
                {'name':'n_bjet',  'threshold':1,      'type':'geq'},
                {'name':'met',    'threshold':230e3,  'type':'geq'},
                {'name':'mt',    'threshold':110e3,  'type':'geq'},
                {'name':'n_lep',  'threshold':1,      'type':'exact'},
                {'name':'lep_pt',  'threshold':25e3,      'type':'geq'}
]

preselection_evaluate = [ 
                {'name':'n_jet',  'threshold':4,      'type':'geq'},
                {'name':'n_bjet',  'threshold':1,      'type':'geq'},
                {'name':'met',    'threshold':230e3,  'type':'geq'},
                {'name':'mt',    'threshold':110e3,  'type':'geq'},
                {'name':'n_lep',  'threshold':1,      'type':'exact'},
                {'name':'lep_pt',  'threshold':25e3,      'type':'geq'}
]

lumi = 140e3

#High level variables

#nvar = [
##        #'ht',
#        'jet_pt[0]',
#        'bjet_pt[0]',
#        'amt2',
#        'mt',
#        'met',
##        #'met_phi',
#        'dphi_met_lep',
##        #'dphi_b_lep_max',
##        #'dphi_jet0_ptmiss',
##        #'met_sig',
#        'met_proj_lep',
#        'ht_sig',
#        'm_bl',
#        'lepPt_over_met'
##        #'dr_bjet_lep',
##        #'mT_blMET', #15vars
##        'n_jet'
##        #'n_bjet'
##        #'jet_pt[1]',
##        #'jet_pt[2]',
##        #'jet_pt[3]',
##        #'lep_pt[0]',
##        #'dr_jet_jet_max',
##        #'ttbar_pt',
##        #'ttbar_dphi',
##        #'m_jet1_jet2',
##        #'m_jet_jet_min',
##        #'m_jet_jet_max'
##        #'tt_cat_TRUTH3'
##        #'n_lep',
##        #'lep_pt',
##        #'lep_eta',
##        #'lep_phi',
##        #'lep_e'
##        #'jet_eta[0]',
##        #'jet_phi[0]',
##        #'dphi_jet1_ptmiss',
##        #'dphi_jet2_ptmiss',
##        #'dphi_jet3_ptmiss',
##        #'dphi_min_ptmiss',
##        #'dphi_b_ptmiss_max'
#        ]

#Low level variables

nvar = [
        'met',
        'met_phi',
        'dphi_met_lep',
        'mt',
        'n_jet',
        'n_bjet',
        #'jet_pt[0]',
        #'jet_eta[0]',
        #'jet_phi[0]',
        #'jet_e[0]',
        #'jet_pt[1]',
        #'jet_eta[1]',
        #'jet_phi[1]',
        #'jet_e[1]',
        #'jet_pt[2]',
        #'jet_eta[2]',
        #'jet_phi[2]',
        #'jet_e[2]',
        #'jet_pt[3]',
        #'jet_eta[3]',
        #'jet_phi[3]',
        #'jet_e[3]',
        #'jet_pt[4]',
        #'jet_eta[4]',
        #'jet_phi[4]',
        #'jet_e[4]',
        #'jet_pt[5]',
        #'jet_eta[5]',
        #'jet_phi[5]',
        #'jet_e[5]',
        #'jet_pt[6]',
        #'jet_eta[6]',
        #'jet_phi[6]',
        #'jet_e[6]',
        #'jet_pt[7]',
        #'jet_eta[7]',
        #'jet_phi[7]',
        #'jet_e[7]',
        #'jet_pt[8]',
        #'jet_eta[8]',
        #'jet_phi[8]',
        #'jet_e[8]',
        #'jet_pt[9]',
        #'jet_eta[9]',
        #'jet_phi[9]',
        #'jet_e[9]',
        'lep_pt[0]',
        'lep_eta[0]',
        'lep_phi[0]',
        'lep_e[0]'
]

weight = [
          'weight',
          'xs_weight',
          'sf_total',
          'weight_sherpa22_njets'
]
