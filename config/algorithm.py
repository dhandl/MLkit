from collections import namedtuple

Algorithm = namedtuple("Algorithm", "name modelname options")

# define your ML algorithm here
analysis = [
Algorithm('NN',
          'DNN_ADAM_layer3x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass',
          {
          'layers':[128,128,128],
          'ncycles':50,
          'batchSize':128,
          'dropout':0.5,
          'optimizer':'adam',
          'activation':'relu',
          'initializer':'normal',
          'regularizer':0.01,
          'learningRate':0.01,
          'decay':0.0,
          'momentum':0.0,
          'nesterov':False,
          'multiclassification':True
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
          'RNN_jetOnly_ADAM_LSTM25_1layer_batch512_GlorotNormalInitializer_dropout0p3',
          {
          'collection':['jet'],
          'unit_type':'LSTM',
          'n_units':25,
          'combinedDim':[256,256,256,256,256],
          'epochs':100,
          'batchSize':512,
          'dropout':0.3, 
          'optimizer':'ADAM',
          'activation':'relu',
          'initializer':'glorot_normal', 
          'learningRate':0.01,
          'decay':0.0, 
          'momentum':0.0,
          'nesterov':False,
          'mergeModels':False, 
          'multiclassification':False
          }
),
]
