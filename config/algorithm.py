from collections import namedtuple

Algorithm = namedtuple("Algorithm", "name modelname options")

# define your ML algorithm here
analysis = [
Algorithm('NN',
          'DNN_ADAM_leakyReLU_layer128_batch32_NormalInitializer_l1-0p01_lr0p001_decay0',
          {
          'layers':[128],
          'ncycles':100,
          'batchSize':32,
          'dropout':0.5,
          'optimizer':'adam',
          'activation':'linear',
          'initializer':'normal',
          'regularizer':0.01,
          'classWeight':'balanced',
          'learningRate':0.001,
          'decay':0.0,
          'momentum':0.0,
          'nesterov':False,
          'multiclassification':False
          }
),
Algorithm('BDT',
          'AdaBoost_d3_mspl0p025_nEst850_lr0p5',
          {
           'classifier':'AdaBoost',
           'max_depth':3,
           'min_samples_leaf':0.025,
           'n_estimators':850,
           'learning_rate':0.5
          }
),
Algorithm('RNN',
          'RNN_jetOnly_ADAM_leakyReLU_LSTM32_128NNlayer_batch32_BatchNorm_NormalInitializer_l2-0p01',
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
          'initializer':'normal',
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
