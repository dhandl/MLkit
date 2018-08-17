from collections import namedtuple

Algorithm = namedtuple("Algorithm", "name modelname options")

# define your ML algorithm here
analysis = [
Algorithm('NN',
          'DNN_ADAM_leakyReLU_layer128_batch32_NormalInitializer_l1-0p01',
          {
          'layers':[128],
          'ncycles':50,
          'batchSize':32,
          'dropout':0.5,
          'optimizer':'adam',
          'activation':'linear',
          'initializer':'normal',
          'regularizer':0.01,
          'classWeight':'sumofweights',
          'learningRate':0.05,
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
          'RNN_jetOnly_ADAM_leakyReLU_GRU32_128NNlayer_batch32_NormalInitializer_l1-0p01',
          {
          'collection':['jet'],
          'unit_type':'GRU',
          'n_units':32,
          'combinedDim':[128,64],
          'epochs':50,
          'batchSize':32,
          'dropout':0.5, 
          'optimizer':'ADAM',
          'activation':'linear',
          'initializer':'normal',
          'regularizer':0.01, 
          'learningRate':0.01,
          'decay':0.0, 
          'momentum':0.0,
          'nesterov':False,
          'mergeModels':True, 
          'multiclassification':False,
          'classWeight':'sumofweights'
          }
),
]
