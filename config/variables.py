
preselection = [ 
                {'name':'n_jet',  'threshold':4,      'type':'geq'},
                {'name':'met',    'threshold':120e3,  'type':'geq'},
                {'name':'mt',    'threshold':90e3,  'type':'geq'},
                {'name':'n_lep',  'threshold':1,      'type':'exact'}
]

lumi = 100e3 

nvar = [
        'n_jet',
        'jet_pt[0]',
        'ht',
        'ttbar_pt',
        'ttbar_dphi',
        'm_jet1_jet2',
        'm_jet_jet_min',
        'm_jet_jet_max'
]

weight = [
          'weight',
          'xs_weight'
]
