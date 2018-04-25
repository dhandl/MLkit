import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import AtlasStyle_mpl

import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_test, y_hat, classes, normalize=False, title='Confusion matrix', fileName=None):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  cm = confusion_matrix(y_test, y_hat, sample_weight=sample_weight)
  np.set_printoptions(precision=3)
  
  cmap = plt.cm.Blues
  
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')
  
  print(cm)
  
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)
  
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
  
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

  AtlasStyle_mpl.ATLASLabel(ax1, 0.02, 0.9, 'Work in progress')

  if fileName:
    plt.savefig(fileName+".pdf")
    plt.savefig(fileName+".png")
    plt.close()