import os, sys
import numpy as np

# keras
from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Masking, GRU, LSTM, Merge, Dense, Dropout, Input, concatenate, Flatten, LeakyReLU
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import initializers
from keras.optimizers import SGD
from keras.utils import np_utils

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

  try:
    model.fit(X_train, y_train)
  except KeyboardInterrupt:
    print '--- Training ended early. ---'
  y_predicted = model.predict(X_test)
  print classification_report(y_test, y_predicted, target_names=["background", "signal"])

  print "BDT finished!"
  return model, y_predicted

def trainNN(X_train, X_test, y_train, y_test, w_train, w_test, netDim, epochs, batchSize, dropout, optimizer, activation, initializer, regularizer, learningRate=0.01, decay=0.0, momentum=0.0, nesterov=False, multiclass = False):
  print "Performing a Deep Neural Net!"

  print 'Standardize training set...'
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  
  class_weight = compute_class_weight('balanced', np.unique(y_train), y_train)
  classes = len(np.bincount(y_train.astype(int)))

  #create 'one-hot' vector for y
  y_train = np_utils.to_categorical(y_train, classes)
  y_test = np_utils.to_categorical(y_test, classes)

  model = Sequential()
  first = True
  if first:
    model.add(Dense(netDim[0], input_dim=X_train.shape[1], activation=activation, kernel_initializer=initializer))
    model.add(BatchNormalization())
    first = False
  for layer in netDim[1:len(netDim)]:
    model.add(Dense(layer, activation=activation, kernel_initializer=initializer, kernel_regularizer=l2(regularizer)))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
  if multiclass:
    model.add(Dense(classes, activation='softmax'))
    loss = 'categorical_crossentropy'
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
  try:
  #history = model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, shuffle=True, class_weight={i:class_weight[i] for i in range(len(class_weight))}, validation_data=(X_test,y_test), callbacks = [EarlyStopping(verbose=True, patience=20)])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, shuffle=True,
              class_weight={i:class_weight[i] for i in range(len(class_weight))},
              sample_weight=w_train, validation_data=(X_test,y_test,w_test),
              callbacks = [EarlyStopping(verbose=True, patience=10, monitor='val_acc')])
  except KeyboardInterrupt:
    print '--- Training ended early ---'
  print 'Testing...'
  score = model.evaluate(X_test, y_test, sample_weight=w_test)
  print("\n%s: %.2f%%" % (model.metrics_names[0], score[0]*100))
  print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
  y_predicted = model.predict(X_test)

  print "DNN finished!"
  return model, history, y_predicted


def trainRNN(X_train, X_test, y_train, y_test, w_train, w_test, sequence, collection, unit_type, n_units, combinedDim, epochs, batchSize, dropout, optimizer, activation, initializer, learningRate=0.01, decay=0.0, momentum=0.0, nesterov=False, mergeModels=False, multiclass=False):
  print "Performing a Deep Recurrent Neural Net!"

  if type(sequence) == list:
    for seq in sequence:
      print 'Prepare channel for {} collection...'.format(seq['name'])
      SHAPE = seq['X_train'].shape[1:]
      seq['input'] = Input(SHAPE)
      seq['channel'] = Masking(mask_value=-999, name=seq['name']+'_masking')(seq['input'])
      if unit_type.lower() == 'lstm':
        seq['channel'] = LSTM(n_units, name=seq['name']+'_lstm')(seq['channel'])
      if unit_type.lower() == 'gru':
        seq['channel'] = GRU(n_units, name=seq['name']+'_gru')(seq['channel'])
      seq['channel'] = Dropout(dropout, name=seq['name']+'_dropout')(seq['channel'])

  if mergeModels:
    print 'Going to merge sequence model with common NN!'
    print 'Standardize training set...'
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model_inputs = Input(shape=(X_train.shape[1], ))
    layer = Dense(n_units, activation=activation, kernel_initializer=initializer)(model_inputs)
    layer = BatchNormalization()(layer)
    layer = Dropout(dropout)(layer)
    
  if mergeModels:
    combined = concatenate([c['channel'] for c in sequence]+[layer])
  else:
    if len(sequence)>1:
      combined = concatenate([c['channel'] for c in sequence])
    else:
      combined = sequence[0]['channel']
    #for layer in combinedDim:
    #  combined = Dense(layer, activation = activation)(combined)
    #  combined = Dropout(dropout)(combined)
  if multiclass:
    combined_output = Dense(len(np.bincount(y_train)), activation='softmax')(combined)
    loss = 'categorical_crossentropy'
  else:
    combined_outputs = Dense(1, activation='sigmoid')(combined)
    loss = 'binary_crossentropy'
  
  if mergeModels:
    combined_rnn = Model(inputs=[seq['input'] for seq in sequence]+[model_inputs], outputs=combined_outputs)
  else:
    if len(sequence)>1:
      combined_rnn = Model(inputs=[seq['input'] for seq in sequence], outputs=combined_outputs)
    else:
      combined_rnn = Model(inputs=sequence[0]['input'], outputs=combined_outputs)

  combined_rnn.summary()
  combined_rnn.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
  print 'Training...'
  class_weight = compute_class_weight('balanced', np.unique(y_train), y_train)
  try:
    if mergeModels:
      history = combined_rnn.fit([seq['X_train'] for seq in sequence]+[X_train], y_train,
                class_weight=class_weight, sample_weight=w_train, epochs=epochs, batch_size=batchSize,
                callbacks = [EarlyStopping(verbose=True, patience=10, monitor='loss')])
                #ModelCheckpoint('./models/combinedrnn_tutorial-progress', monitor='val_loss', verbose=True, save_best_only=True)
    else:
      history = combined_rnn.fit([seq['X_train'] for seq in sequence], y_train,
                class_weight=class_weight, sample_weight=w_train, epochs=epochs, batch_size=batchSize,
                callbacks = [EarlyStopping(verbose=True, patience=10, monitor='acc')])
  except KeyboardInterrupt:
      print 'Training ended early.'

  print 'Testing...'
  if mergeModels:
    score = combined_rnn.evaluate([seq['X_test'] for seq in sequence]+[X_test], y_test, batch_size=batchSize)
    y_predicted = combined_rnn.predict([seq['X_test'] for seq in sequence]+[X_test], batch_size=batchSize)
  else:
    if len(seq)>1:
      score = combined_rnn.evaluate([seq['X_test'] for seq in sequence], y_test)
      y_predicted = combined_rnn.predict([seq['X_test'] for seq in sequence], batch_size=batchSize)
    else:
      score = combined_rnn.evaluate(sequence[0]['X_test'], y_test)
      y_predicted = combined_rnn.predict(sequence[0]['X_test'], batch_size=batchSize)
  #print("\n%s: %.2f%%" % (combined_rnn.metrics_names[0], score[0]*100))
  #print("\n%s: %.2f%%" % (combined_rnn.metrics_names[1], score[1]*100))
  

  print "RNN finished!"
  return combined_rnn, history, y_predicted

def trainCNN():
  x = Input(shape=(images_train.shape[1:]))
  h = Conv2D(32, kernel_size=7, strides=1)(x)
  h = LeakyReLU()(h)
  h = Dropout(0.2)(h)
  
  h = Conv2D(64, kernel_size=7, strides=1)(h)
  h = LeakyReLU()(h)
  h = Dropout(0.2)(h)
  
  h = Conv2D(128, kernel_size=5, strides=1)(h)
  h = LeakyReLU()(h)
  h = Dropout(0.2)(h)
  
  h = Conv2D(256, kernel_size=5, strides=1)(h)
  h = LeakyReLU()(h)
  h = Flatten()(h)
  h = Dropout(0.2)(h)
  y = Dense(1, activation='sigmoid')(h)
  
  cnn_model = Model(x, y)
  cnn_model.compile('adam', 'binary_crossentropy', metrics=['acc'])
  
  cnn_model.summary()
  
  # cnn_model.fit(
  #     images_train, labels_train,
  #     epochs=100,
  #     batch_size=512,
  #     validation_data=(images_val, labels_val),
  #     callbacks=[
  #         EarlyStopping(verbose=True, patience=30, monitor='val_loss'),
  #         ModelCheckpoint('./models/cnn-model.h5', monitor='val_loss',
  #                         verbose=True, save_best_only=True)
  #     ]
  # )

  return cnn, history, y_predicted
