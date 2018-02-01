import os, sys
import numpy as np

# keras
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import initializers
from keras.optimizers import SGD

# scikit-learn
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.externals import joblib
from sklearn.utils.class_weight import compute_class_weight

def trainBDT(X_train, X_test, y_train, y_test, w_train, w_test, classifier, max_depth, min_samples_leaf, n_estimators, learning_rate):
  print "Performing a Boosted Decision Tree!"
  if classifier.lower() == 'adaboost':
    print "Using AdaBoost technique!"
    dt = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    model = AdaBoostClassifier(dt, algorithm="SAMME.R", n_estimators=n_estimators, learning_rate=learning_rate)
  elif classifier.lower() == 'gradientboost':
    print "Using GradientBoost technique!"
    model = GradientBoostingClassifier(max_depth=max_depth, criterion='friedman_mse', min_samples_leaf=min_samples_leaf, loss='deviance', n_estimators=n_estimators, learning_rate=learning_rate, warm_start=True)

  model.fit(X_train, y_train, sample_weight=w_train)
  y_predicted = model.predict(X_test)
  print classification_report(y_test, y_predicted, target_names=["background", "signal"])

  print "BDT finished!"
  return model, y_predicted

def trainNN(X_train, X_test, y_train, y_test, w_train, w_test, netDim, epochs, batchSize, dropout, optimizer, activation, initializer,learningRate=0.01, decay=0.0, momentum=0.0, nesterov=False, multiclass = False):              
  print "Performing a Deep Neural Net!"
  model = Sequential()
  first = True
  if first:
    model.add(Dense(netDim[0], input_dim=X_train.shape[1], activation=activation, kernel_initializer=initializer))
    #model.add(Dropout(0.4))
    model.add(BatchNormalization())
    first = False
  for layer in netDim[1:len(netDim)-1]:
    model.add(Dense(layer, activation=activation, kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
  if multiclass:
    model.add(Dense(len(np.bincount(y_train)), activation='softmax'))
    loss = 'sparse_categorical_crossentropy'
  else:
    model.add(Dense(1, activation='sigmoid'))
    loss = 'binary_crossentropy'

  # Set loss and optimizer
  if optimizer.lower() == 'sgd':
    print 'Going to use stochastic gradient descent method for learning!'
    optimizer = SGD(lr=learningRate, decay=decay, momentum=momentum , nesterov=True)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
  else:
    print 'Going to use %s as optimizer. Learning rate, decay and momentum will not be used during training!'%(optimizer)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

  print model.summary()
  print "Training..."
  # if using an unbalanced set of samples sci-kit learns compute_class_weight fct. equally penalizes under/over represented classes in the training set
  # n_samples / (n_classes * np.bincount(y))
  class_weight = compute_class_weight('balanced', np.unique(y_train), y_train)
  #y_train_binarize= label_binarize(y_train, classes=[0,1,2,3])
  model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, shuffle=True, class_weight={i:class_weight[i] for i in range(len(class_weight))}, validation_data=(X_test,y_test), callbacks = [EarlyStopping(verbose=True, patience=10)])
  # TODO: add callbacks and ModelCheckpoint
  # Store model to file
  print 'Testing...'
  score = model.evaluate(X_test, y_test)
  print("\n%s: %.2f%%" % (model.metrics_names[0], score[0]*100))
  print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
  y_predicted = model.predict(X_test)

  print "DNN finished!"
  return model, y_predicted


def trainRNN():
  # first test of a recurrent neural network
  return model, y_predicted
