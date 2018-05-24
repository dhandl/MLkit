import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import prepareTraining as pT
from collections import namedtuple
Sample = namedtuple('Sample', 'name dataframe')

def draw_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,save=False,fileName="Test",isTrain=False, addStr = ''):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    cm is the confusion matrix
    """
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
                 color="white" if cm[i, j] > thresh else "black", fontsize=14)

    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    
    fig = plt.gcf()
    fig.set_size_inches(9., 7.)
    
    if isTrain:
        extraStr = 'Train'
    else:
        extraStr = ''
        
    if save:
        if not os.path.exists("./plots/"):
            os.makedirs("./plots/")
            print("Creating folder plots")
        plt.savefig("plots/"+fileName+"_ConfusionMatrix" + addStr + extraStr +".pdf")
        plt.savefig("plots/"+fileName+"_ConfusionMatrix" + addStr + extraStr +".png")
        plt.close()
    
    
def plot_confusion_matrix(y_true, y_predict, filename="Test",save=False,isTrain=False, addStr=''):
    """
    Plotting (and printing) the confusion matrix
    """
    print('Plotting the confusion matrix...')
    yhat_cls = np.argmax(y_predict, axis=1)
    cnf_matrix = confusion_matrix(y_true, yhat_cls)
    np.set_printoptions(precision=2)
    
    draw_confusion_matrix(cnf_matrix, classes=[r'Signal', r'$t\overline{t}$', 'Single Top', r'$W$ + jets'],
                      normalize=True,
                      title='Normalized Confusion Matrix',save=save,fileName=filename,isTrain=isTrain, addStr=addStr)
    
def plot_confusion_matrix_datapoint(SignalList, model, preselection, nvar, weight, lumi, save=False, fileName='Test', multiclass=True):
    
    '''
    Evaluate the confusion matrix on certain datapoints. sigList is supposed to be in a form like in config/samples.py
    '''
    
    print '----- Plotting the confusion matrices for different datapoints-----'
    
    met_cut = False
    for pre in preselection:
        if pre['name'] == 'met':
            met_cut = True
    
    if not met_cut:
        print 'Using no met-preselection!'
    
    input='/project/etp5/dhandl/samples/SUSY/Stop1L/hdf5/cut_mt30_met60_preselection/'
    
    bkgList = [
    {'name':'powheg_ttbar', 'path':input+'powheg_ttbar/'},
    {'name':'powheg_singletop', 'path':input+'powheg_singletop/'},
    {'name':'sherpa22_Wjets', 'path':input+'sherpa22_Wjets/'}
    ]
    
    #Loading background once
    print 'Loading background...'
    
    Background = []
    for b in bkgList:
      print 'Loading background {} from {}...'.format(b['name'], b['path'])
      Background.append(Sample(b['name'], pT.loadDataFrame(b['path'], preselection, nvar, weight, lumi)))
      
    bkg = np.empty([0, Background[0].dataframe[0].shape[1]])
    bkg_w = np.empty(0)
    bkg_y = np.empty(0)
    
    for i, b in enumerate(Background):
      i = i + 1
      bkg = np.concatenate((bkg, b.dataframe[0]))
      bkg_w = np.concatenate((bkg_w, b.dataframe[1]))
      bkg_y = np.concatenate((bkg_y, np.full(b.dataframe[0].shape[0], i)))
      
    #Evaluating on signal for each set of points
    print 'Evaluating on signal sets...'
    
    for sigList in SignalList:
        Signal = []
        addStr = '_stop_bWN_'
        name=False
        for s in sigList:
            if not name:
                addStr += s['name'].replace('stop_bWN_', '')
                name=True
            else:
                addStr += s['name'].replace(s['name'][:12], '')
            
            print 'Loading signal {} from {}...'.format(s['name'], s['path'])
            Signal.append(Sample(s['name'], pT.loadDataFrame(s['path'], preselection, nvar, weight, lumi)))
        
        sig = np.empty([0, Signal[0].dataframe[0].shape[1]])
        sig_w = np.empty(0)
        sig_y = np.empty(0)
    
        for s in Signal:
            sig = np.concatenate((sig, s.dataframe[0]))
            sig_w = np.concatenate((sig_w, s.dataframe[1]))
            sig_y = np.concatenate((sig_y, np.zeros(s.dataframe[0].shape[0])))
      
        X = np.concatenate((sig, bkg))
        w = np.concatenate((sig_w, bkg_w))
  
        if multiclass:
            y = np.concatenate((sig_y, bkg_y))
        else:
            y = []
            for _df, ID in [(sig, 0), (bkg, 1)]:
                y.extend([ID] * _df.shape[0])
            y = np.array(y)

        scaler=StandardScaler()

        X_scaled = scaler.fit_transform(X)
        y_predict = model.predict(X_scaled)
        y_true = y
        
        if not met_cut:
            addStr += '_no_met_cut'

        plot_confusion_matrix(y_true, y_predict, filename=fileName, save=save, addStr=addStr)