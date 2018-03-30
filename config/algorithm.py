from collections import namedtuple

Algorithm = namedtuple("Algorithm", "name modelname options")

# define your ML algorithm here
analysis = [
Algorithm('NN',
          'DNN_ADAM_layer5x64_batch16_GlorotNormalInitializer_dropout0p2',
          {
          'layers':[64,64,64,64,64],
          'ncycles':100,
          'batchSize':16,
          'dropout':0.2,
          'optimizer':'ADAM',
          'activation':'relu',
          'initializer':'glorot_normal',
          'regularizer':0.01,
          'learningRate':0.01,
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
