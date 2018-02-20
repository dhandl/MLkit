from collections import namedtuple

Algorithm = namedtuple("Algorithm", "name modelname options")

# define your ML algorithm here
analysis = [
Algorithm('NN',
          'DNN_ADAM_layer128-64_batch16',
          {
          'layers':[128,64],
          'ncycles':100,
          'batchSize':16,
          'dropout':0.3,
          'optimizer':'adam',
          'activation':'relu',
          'initializer':'glorot_uniform',
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
