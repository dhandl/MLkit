
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
        'jet_pt[3]'
]

weight = [
          'weight',
          'xs_weight'
]
