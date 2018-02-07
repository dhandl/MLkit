import os, sys
import numpy as np

# keras
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Masking, GRU, LSTM, Merge
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

def trainBDT(X_train, X_test, y_train, y_test, w_train, w_test, classifier, max_depth, min_samples_leaf, n_estimators, learning_rate, reproduce=False):
  print "Performing a Boosted Decision Tree!"
  if reproduce:
    print "Warning! Constant seed is activated"
    random_state=14
  if classifier.lower() == 'adaboost':
    print "Using AdaBoost technique!"
    dt = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)
    model = AdaBoostClassifier(dt, algorithm="SAMME.R", n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
  elif classifier.lower() == 'gradientboost':
    print "Using GradientBoost technique!"
    model = GradientBoostingClassifier(max_depth=max_depth, criterion='friedman_mse', min_samples_leaf=min_samples_leaf, loss='deviance', n_estimators=n_estimators, learning_rate=learning_rate, warm_start=True)

  model.fit(X_train, y_train)
  y_predicted = model.predict(X_test)
  print classification_report(y_test, y_predicted, target_names=["background", "signal"])

  print "BDT finished!"
  return model, y_predicted

def trainNN(X_train, X_test, y_train, y_test, w_train, w_test, netDim, epochs, batchSize, dropout, optimizer, activation, initializer,learningRate=0.01, decay=0.0, momentum=0.0, nesterov=False, multiclass = False):
  print "Performing a Deep Neural Net!"

  print 'Standardize training set...'
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  model = Sequential()
  first = True
  if first:
    model.add(Dense(netDim[0], input_dim=X_train.shape[1], activation=activation, kernel_initializer=initializer))
    model.add(BatchNormalization())
    first = False
  for layer in netDim[1:len(netDim)]:
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
  class_weight = compute_class_weight('balanced', np.unique(y_train), y_train)
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


#def trainRNN(X_train, X_test, y_train, y_test, w_train, w_test, nvar, collection, multiclass=False):
#
#  if type(collection) == list:
#    sequences = []
#    for seq in collection:
#      print 'Prepare sequence for {} collection...'.format(seq)
#      sequences.append([key for key in .keys() if key.startswit(seq)])
# netDim, epochs, batchSize, dropout, optimizer, activation, initializer,learningRate=0.01, decay=0.0, momentum=0.0, nesterov=False):
#  return model, y_predicted
