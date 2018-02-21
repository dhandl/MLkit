from collections import namedtuple

Algorithm = namedtuple("Algorithm", "name modelname options")

# define your ML algorithm here
analysis = [
Algorithm('NN',
          'DNN_rmsprop_layer128_batch512_GlorotNormalInitializer',
          {
          'layers':[128],
          'ncycles':100,
          'batchSize':512,
          'dropout':0.3,
          'optimizer':'rmsprop',
          'activation':'relu',
          'initializer':'glorot_normal',
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
]
