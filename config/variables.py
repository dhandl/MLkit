
preselection = [ 
                {'name':'n_jet',  'threshold':4,      'type':'geq'},
                {'name':'met',    'threshold':120e3,  'type':'geq'}
]

lumi = 100e3 

nvar = [
        'met',
        'n_jet',
        'jet_pt[0]',
        'jet_pt[1]',
        'jet_pt[2]',
        'jet_pt[3]',
        'jet_eta[0]',
        'jet_eta[1]',
        'jet_eta[2]',
        'jet_eta[3]',
        'jet_phi[0]',
        'jet_phi[1]',
        'jet_phi[2]',
        'jet_phi[3]',
        'jet_deltaRj[0]',
        'jet_deltaRj[1]',
        'jet_deltaRj[2]',
        'jet_deltaRj[3]',
        'dphi_jet0_ptmiss',
        'dphi_jet1_ptmiss',
        'dphi_jet2_ptmiss',
        'dphi_jet3_ptmiss',
        'dr_jet_jet_min',
        'dr_jet_jet_max',
        'deta_jet_jet_min',
        'deta_jet_jet_max',
        'dphi_jet_jet_min',
        'dphi_jet_jet_max',
        'm_jet1_jet2',
        'm_jet_jet_min',
        'm_jet_jet_max'
]

weight = [
          'weight',
          'xs_weight'
]
