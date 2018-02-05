from collections import namedtuple

Algorithm = namedtuple("Algorithm", "name modelname options")

# define your ML algorithm here
analysis = [
Algorithm('NN',
          'DNN_ADAM_layer16-16_epochs50_batch512',
          {
          'layers':[16,16],
          'ncycles':50,
          'batchSize':512,
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
)
]
