from collections import namedtuple

Algorithm = namedtuple("Algorithm", "name modelname options")

# define your ML algorithm here
analysis = [
Algorithm('NN',
          'DNN_ADAM_leakyReLU_layer256_batch32_GlorotNormalInitializer_l1-0p01_lr0p0001_decay0p0',
          {
          'layers':[256],
          'ncycles':1000,
          'batchSize':32,
          'dropout':0.5,
          'optimizer':'adam',
          'activation':'linear',
          'initializer':'glorot_normal',
          'regularizer':0.01,
          'classWeight':'balanced',
          'learningRate':0.0001,
          'decay':0.0,
          'momentum':0.0,
          'nesterov':False,
          'multiclassification':False
          }
),
Algorithm('BDT',
          'XGBoost_d3_nTrees1000_lr0p1',
          {
           'classifier':'xgboost',
           'learning_rate':0.1,
           'max_depth':3,
           'n_estimators':1000,
           'lambda':1, # Default=1, L2 regularization term on weights (analogous to Ridge regression)
           'alpha':0, # Default=0, L1 regularization term on weight (analogous to Lasso regression)
           'gamma':0, # A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split.
          'scale_pos_weights':1 # Default=1, A value greater than 0 should be used in case of high class imbalance as it helps in faster convergence.
          }
),
Algorithm('RNN',
          'RNN_jetOnly_ADAM_leakyReLU_LSTM32_128NNlayer_batch32_BatchNorm_GlorotNormalInitializer_l2-0p01_lr0p001_decay0p0',
          {
          'collection':['jet'],
          'removeVar':['_m', '_mv2c10', '_id', '0_pt', '0_eta', '0_phi', '0_e', '1_pt', '1_eta', '1_phi', '1_e'],
          'unit_type':'LSTM',
          'n_units':32,
          'combinedDim':[128],
          'epochs':25,
          'batchSize':32,
          'dropout':0.5, 
          'optimizer':'ADAM',
          'activation':'linear',
          'initializer':'glorot_normal',
          'regularizer':0.01,
          'learningRate':0.001,
          'decay':0.0, 
          'momentum':0.0,
          'nesterov':False,
          'mergeModels':True, 
          'multiclassification':False,
          'classWeight':'balanced'
          }
),
]
