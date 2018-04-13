
preselection = [ 
                {'name':'n_jet',  'threshold':4,      'type':'geq'},
                {'name':'n_bjet',  'threshold':1,      'type':'geq'},
                {'name':'met',    'threshold':100e3,  'type':'geq'},
                {'name':'mt',    'threshold':90e3,  'type':'geq'},
                {'name':'n_lep',  'threshold':1,      'type':'exact'}
                #{'name':'tt_cat_TRUTH3',  'threshold':0,      'type':'geq'},
                #{'name':'tt_cat_TRUTH3',  'threshold':6,      'type':'leq'},
                #{'name':'tt_cat_TRUTH3',  'threshold':1,      'type':'not'},
                #{'name':'tt_cat_TRUTH3',  'threshold':4,      'type':'not'}
]

lumi = 140e3 

nvar = [
        'n_jet',
        'ht',
        'jet_pt[0]',
        'bjet_pt[0]',
        'jet_pt[1]',
        'jet_pt[2]',
        'jet_pt[3]',
        'lep_pt[0]',
        'amt2',
        'mt',
        'met',
        'dphi_met_lep',
        'met_sig',
        'ht_sig',
        'm_bl',
        'dr_bjet_lep',
        'mT_blMET'
        #'dr_jet_jet_max',
        #'ttbar_pt',
        #'ttbar_dphi',
        #'m_jet1_jet2',
        #'m_jet_jet_min',
        #'m_jet_jet_max'
        #'tt_cat_TRUTH3'
]

weight = [
          'weight',
          'xs_weight',
          'sf_total',
          'weight_sherpa22_njets'
]
