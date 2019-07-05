
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

nvar = [
        #'met', 'met_phi', 'mt', 'dphi_met_lep', 'm_bl', 'bjet_pt[0]', 'n_jet', 'n_bjet', 'lep_pt[0]', 'lep_eta[0]', 'lep_phi[0]', 'lep_eta[0]' 
        #'met',
        'met_phi',
        'mt',
        'dphi_met_lep',
        'm_bl',
        'bjet_pt[0]',
        'n_jet',
        'n_bjet',
        'lep_pt[0]',
        'lep_eta[0]',
        'lep_phi[0]',
        'lep_e[0]'
        ]

#Low level variables
#nvar = [
#        'met',
#        'met_phi',
#        'mt',
#        'dphi_met_lep',
#        'm_bl',
#        'bjet_pt[0]',
#        'n_jet',
#        'n_bjet',
#        'lep_pt[0]',
#        'lep_eta[0]',
#        'lep_phi[0]',
#        'lep_e[0]'
#        ]


weight = [
          'weight',
          'lumi_weight',
          'xs_weight',
          'sf_total'
]
