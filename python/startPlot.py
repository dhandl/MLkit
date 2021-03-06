import os,sys
from sklearn.preprocessing import StandardScaler
import h5py
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import pickle
import numpy as np

sys.path.append('./python/plotting/')

import plot_TrainTest_score
import plot_ConfusionMatrix
import plot_Classification
#import plot_Classification2
import plot_learning_curve
import plot_output_score
import plot_output_score_multiclass
import plot_output_score2D
import plot_piechart
#import evaluate_signalGrid
import plot_ROCcurves
import plot_Correlation

def startPlot(modelDir, binning=[50,0,1.], save=False, multiclass=True):
    '''
    Plot all important things
        
    - modelDir: Directory of model
    
    - binning = [bins, start, stop] default: [50,0,1.]
    
    - save: Save Files in ./plots/ (True/False)
    '''
    #Load models
    
    print 'Loading infos from infofile...'
    
    infofile = open(modelDir.replace('.h5','_infofile.txt'))
    infos = infofile.readlines()
   
    analysis=infos[0].replace('Used analysis method: ','').replace('\n','')
    parameters=infos[1].replace('Used parameters for this analysis algorithm: {','').replace('}\n','').split(',')
    variables=infos[5].replace('Used variables for training: ','').replace('\n','').split()
    weights=infos[6].replace('Used weights: ', '').replace('\n','').split()
    lumi=float(infos[8].replace('Used Lumi: ','').replace('\n',''))
    
    preselection_raw=infos[7].replace('Used preselection: ', '').replace('; \n', '').split(';')
    preselection=[]
    for x in preselection_raw:
        xdict = {}
        xdict['name']= x.split()[0].split('-')[0]
        xdict['threshold']= float(x.split()[1])
        xdict['type'] = x.split()[3]
        if xdict['type'] == 'condition':
            xdict['variable'] = x.split()[5]
            xdict['lessthan'] = float(x.split()[7])
            xdict['morethan'] = float(x.split()[10])
        preselection.append(xdict)
        
    print 'Loading dataset...'
    datasetDir = 'TrainedModels/datasets/' + infos[3].replace('Used dataset: ', '').replace('\n','') + '.h5'
    
    dataset = h5py.File(datasetDir)
    
    filenames = modelDir.replace('TrainedModels/models/','').replace('.h5','')
    #filenames = 'blabla'
 
    print 'Using dataset from:', datasetDir
    
    print 'Loading model...'
    
    if analysis.lower() == 'rnn':
      print('Recurrent Neural Network detected!')
      pickleDir = modelDir.replace('.h5', '_history.pkl')
      model = load_model(modelDir)
      model.load_weights(modelDir.replace('.h5' , '_weights.h5').replace('models' , 'weights'))
    elif analysis.lower() == 'nn':
      print('Neural Network detected!')
      pickleDir = modelDir.replace('.h5', '_history.pkl')
      model = load_model(modelDir)
      model.load_weights(modelDir.replace('.h5' , '_weights.h5').replace('models' , 'weights'))

    elif analysis.lower() == 'bdt':
      print('Boosted Decision Tree detected!')
      model = joblib.load(modelDir)
    
    #Get the data and scale it, if necessary
    print 'Loading data from dataset...'
    
    X_train = dataset['X_train'][:]
    X_test = dataset['X_test'][:]
    X = dataset['X'][:]
    y_train= dataset['y_train'][:]
    y_test= dataset['y_test'][:]
    y = dataset['y'][:]
    w = dataset['w'][:]

    if analysis == 'rnn':
      sequence = []
      seq = getSequence(parameters)
      for idx, c in enumerate(seq):
        sequence.append({'name':c, 'X_train':dataset['X_train_'+c][:], 'X_test':dataset['X_test_'+c][:]})

    if os.path.exists(modelDir.replace('.h5' , '_scaler.pkl')):
      scaler = joblib.load(modelDir.replace('.h5' , '_scaler.pkl')) # RNN sequences already scaled!
      X_train_scaled = scaler.transform(X_train)
      X_test_scaled = scaler.transform(X_test)
      X_scaled = scaler.transform(X)
    else:
      X_train_scaled = X_train
      X_test_scaled = X_test
      X_scaled = X
    
    if analysis == 'rnn':
      y_predict_train = model.predict([s['X_train'] for s in sequence] + [X_train_scaled])
      y_predict_test = model.predict([s['X_test'] for s in sequence] + [X_test_scaled])
      y_predict = np.concatenate((y_predict_train, y_predict_test))
      y = np.concatenate((y_train, y_test))
    elif analysis == 'nn':
      y_predict_train = model.predict(X_train_scaled)
      y_predict_test = model.predict(X_test_scaled)
      y_predict = model.predict(X_scaled)
    elif analysis == 'bdt':
      y_predict_train = model.predict_proba(X_train_scaled)
      y_predict_test = model.predict_proba(X_test_scaled)
      y_predict = model.predict_proba(X_scaled)
    
    sig_predicted_train = y_predict_train[y_train==0]
    sig_predicted_test = y_predict_test[y_test==0]
    sig_w_train = dataset['w_train'][y_train==0]
    sig_w_test = dataset['w_test'][y_test==0]
    bkg_predicted_train = y_predict_train[y_train!=0]
    bkg_predicted_test = y_predict_test[y_test!=0]
    bkg_w_train = dataset['w_train'][y_train!=0]
    bkg_w_test = dataset['w_test'][y_test!=0]
    
    sig_predicted = y_predict[y==0]
    sig_w = dataset['w'][y==0]
    bkg_predicted= y_predict[y!=0]
    bkg_w = dataset['w'][y!=0]
   
    print  y_predict.shape, y_predict[:5]
    #outputScore = y_predict[:,0]
    
#    vars_2D = [
#            {'x':'met','y':'mt','xlabel':r'$E_{T}^{miss}$','ylabel':r'$m_{T}$', 'xbinning':[100,1000,80], 'ybinning':[90,700,80], 'xmev':True, 'ymev':True},
#            {'x':'met_proj_lep', 'y':'dphi_met_lep','xlabel':r'$E_{T,l}^{miss}$', 'ylabel':r'$\Delta\Phi(l, E_{T}^{miss})$', 'xbinning':[90,700,100], 'ybinning': [0,3.2,100], 'xmev':True, 'ymev':False},
#            {'x':'met','y':'jet_pt[0]','xlabel':r'$E_{T}^{miss}$', 'ylabel':r'$p_{T}^{jet0}$', 'xbinning':[100,1000,80], 'ybinning': [25,1000,80], 'xmev':True, 'ymev':True},
#            {'x':'amt2','y':'m_bl', 'xlabel':r'$am_{T2}$ [GeV]', 'ylabel':r'$m_{b,l}$', 'xbinning':[90,700,100], 'ybinning': [0,700,100], 'xmev':False, 'ymev':True},
#            {'x':'dr_bjet_lep','y':'m_bl', 'xlabel':r'$\Delta R(b,l)$', 'ylabel':r'$m_{b,l}$', 'xbinning':[0,3.2,100], 'ybinning': [0,1000,100], 'xmev':False, 'ymev':True},
#            {'x':'met', 'y':'ht','xlabel':r'$E_{T}^{miss}$', 'ylabel':r'$h_{T}$', 'xbinning':[100,1000,80], 'ybinning': [100,1000,80], 'xmev':True, 'ymev':True},
#            {'y':'mt', 'x':'ht','ylabel':r'$m_{T}$', 'xlabel':r'$h_{T}$', 'ybinning':[90,700,80], 'xbinning': [100,1000,80], 'ymev':True, 'xmev':True},
#            {'x':'met','y':'bjet_pt[0]','xlabel':r'$E_{T}^{miss}$','ylabel':r'$p_{T}^{bjet0}$', 'xbinning':[100,1000,80], 'ybinning':[25,1000,80], 'xmev':True, 'ymev':True},
#            {'x':'met_proj_lep','y':'mt','xlabel':r'$E_{T,l}^{miss}$','ylabel':r'$m_{T}$', 'xbinning':[100,1000,80], 'ybinning':[90,700,80], 'xmev':True, 'ymev':True}
#            ]
    
    #Do various plots
    print 'Start plotting...'
    
    #TrainTestScore for cuts that are used for training
    plot_TrainTest_score.plot_TrainTest_score(sig_predicted_train[:,0], sig_predicted_test[:,0], sig_w_train, sig_w_test, bkg_predicted_train[:,0], bkg_predicted_test[:,0], bkg_w_train, bkg_w_test, binning, normed=1,save=save,fileName=filenames)
    
    #CMs for cuts that are used for training
    plt.figure()
    plot_ConfusionMatrix.plot_confusion_matrix(y_test, y_predict_test, filename=filenames, save=save, isTrain=False)
    
    plt.figure()
    plot_ConfusionMatrix.plot_confusion_matrix(y_train, y_predict_train, filename=filenames, save=save, isTrain=True)
    
    if 'nn' in analysis.lower():
      #Learning Curve
      plt.figure()
      plot_learning_curve.learning_curve_for_keras(pickleDir, save=save, filename=filenames)
    
    #Pie Charts
    plt.figure()
    #plot_piechart.plot_pie_chart(y, y_predict, w, fileName=filenames, save=save)
    
    #Output Score for cuts that are used in training
    plt.figure()
    plot_output_score.plot_output_score(sig_predicted[:,0], sig_w, bkg_predicted[:,0], bkg_w, binning, save=save, fileName=filenames, log=True)
    plot_output_score.plot_output_score(sig_predicted[:,0], sig_w, bkg_predicted[:,0], bkg_w, binning, save=save, fileName=filenames, log=False)    
    
    plt.figure()
    plot_ROCcurves.plot_ROC(y_train, y_test, y_predict_train, y_predict_test, save=save, fileName=filenames)
 
    #plt.figure()
    #plot_Correlation.plotCorrelation(X, y_predict, y, variables, fileName=filenames, save=save, multiclass=multiclass)
    #plot_Correlation.plotCorrelation(X, y_predict, y, variables, fileName=filenames, save=save, plotEPD=False)

    if multiclass:    
      #Classification Plots    
      plt.figure()
      plot_Classification.plot_classification(y_test, y_predict_test, w, fileName=filenames, save=save)
      
      plt.figure()
      plot_Classification.plot_classification(y, y_predict, w, fileName=filenames, save=save, weighted=True)
      
      plt.figure()
      plot_Classification.plot_classification(y_train, y_predict_train, w, fileName=filenames, save=save, train=True)
      
      #plt.figure()
      #plot_Classification2.plot_classification_2(y_test, y_predict_test, fileName=filenames, save=save)
      
      #plt.figure()
      #plot_Classification.plot_classification_datapoint(Signal1, model, preselection, variables, weights, lumi, save=save, fileName=filenames, multiclass=True)    
    
      #bkg1_predicted_test = y_predict_test[y_test==1]
      #bkg1_w_test = dataset['w_test'][y_test==1]
      #bkg2_predicted_test = y_predict_test[y_test==2]
      #bkg2_w_test = dataset['w_test'][y_test==2]
      #bkg3_predicted_test = y_predict_test[y_test==3]
      #bkg3_w_test = dataset['w_test'][y_test==3]
      bkg1_predicted = y_predict[y==1]
      bkg1_w = dataset['w'][y==1]
      bkg2_predicted = y_predict[y==2]
      bkg2_w = dataset['w'][y==2]
      bkg3_predicted = y_predict[y==3]
      bkg3_w = dataset['w'][y==3]

      plot_output_score_multiclass.plot_output_score_multiclass(sig_predicted[:,0], sig_w, bkg1_predicted[:,0], bkg1_w, bkg2_predicted[:,0], bkg2_w, bkg3_predicted[:,0], bkg3_w, bkg_predicted[:,0], bkg_w, binning, save=save, fileName=filenames, log=True)
      plot_output_score_multiclass.plot_output_score_multiclass(sig_predicted[:,0], sig_w, bkg1_predicted[:,0], bkg1_w, bkg2_predicted[:,0], bkg2_w, bkg3_predicted[:,0], bkg3_w, bkg_predicted[:,0], bkg_w, binning, save=save, fileName=filenames, log=False)
    
    #evaluate_signalGrid.evaluate_signalGrid(modelDir, save=save, fileName=filenames)
    #evaluate_signalGrid.evaluate_signalGridCuts(modelDir, save=save, fileName=filenames)
    
def startPlotDataset(modelDir, datasetDir, binning=[50,0,1.], save=False):
    '''
    Plot all important things for specific dataset, which is not the one used for training
        
    - modelDir: Directory of model
    
    - datasetDir: Directory of dataset
    
    - binning = [bins, start, stop] default: [50,0,1.]
    
    - save: Save Files in ./plots/ (True/False)
    '''
    #Load models
    
    print('Loading dataset...')
    
    dataset = h5py.File(datasetDir)
    
    filenames = modelDir.replace('TrainedModels/models/','').replace('.h5','') + '_differentDataset'
    
    print('Using dataset from:', datasetDir)
    
    print('Loading model...')
    
    try:
        pickleDir = modelDir.replace('.h5', '_history.pkl')
        model = load_model(modelDir)
        model.load_weights(modelDir.replace('.h5' , '_weights.h5').replace('models' , 'weights'))
        print('Neural Network detected!')
        print('Scaling and reading values...')
    except IOError:
        model = joblib.load(modelDir)
        print('Boosted Decision Tree detected!')
        return 0
    
    #Get the data and scale it, if necessary
    
    X = dataset['X'][:]
    y = dataset['y'][:]
    
    scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X)
    
    y_predict = model.predict(X_scaled)
    
    sig_predicted = y_predict[y==0]
    sig_w = dataset['w'][y==0]
    
    bkg_predicted= y_predict[y!=0]
    bkg_w = dataset['w'][y!=0]
    
    #Do various plots
    
    evaluate_signalGrid.evaluate_signalGridCut(modelDir, save=save, fileName=filenames)
    
def main():
    print '---------- Warning: startPlot in main ----------'
    modelDir = 'TrainedModels/models/2018-05-17_10-44_DNN_ADAM_layer4x128_batch100_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir1 = 'TrainedModels/models/2018-05-18_15-33_DNN_ADAM_layer3x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir2 = 'TrainedModels/models/2018-05-18_15-04_DNN_ADAM_layer1x80_batch40_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir3 = 'TrainedModels/models/2018-05-18_15-12_DNN_ADAM_layer1x80_batch40_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir4 = 'TrainedModels/models/2018-05-18_15-26_DNN_ADAM_layer3x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir5 = 'TrainedModels/models/2018-05-11_13-34_DNN_ADAM_layer4x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir6 = 'TrainedModels/models/2018-05-28_14-44_DNN_ADAM_layer3x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir7 = 'TrainedModels/models/2018-05-28_16-11_DNN_ADAM_layer1x10_batch10_NormalInitializer_dropout0p5_l2-0p01_multiclass_TEST.h5'
    modelDir8 = 'TrainedModels/models/2018-05-28_17-57_DNN_ADAM_layer3x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir9 = 'TrainedModels/models/2018-05-29_11-33_DNN_SGD_layer3x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir10= 'TrainedModels/models/2018-05-29_10-59_DNN_ADAM_layer3x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir11= 'TrainedModels/models/2018-05-29_17-14_DNN_ADAM_layer3x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir12= 'TrainedModels/models/2018-06-04_13-23_DNN_ADAM_layer1x80_batch40_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir13= 'TrainedModels/models/2018-06-06_11-45_DNN_ADAM_layer3x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir14= 'TrainedModels/models/2018-06-06_11-45_DNN_ADAM_layer3x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir15= 'TrainedModels/models/2018-06-06_11-56_DNN_ADAM_layer3x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir16= 'TrainedModels/models/2018-06-06_12-18_DNN_ADAM_layer3x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir17= 'TrainedModels/models/2018-06-06_12-19_DNN_ADAM_layer3x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir18= 'TrainedModels/models/2018-06-07_13-23_DNN_ADAM_layer3x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir19= 'TrainedModels/models/2018-06-07_14-32_DNN_ADAM_layer3x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir20= 'TrainedModels/models/2018-06-07_18-04_DNN_ADAM_layer3x32_batch64_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir21= 'TrainedModels/models/2018-06-08_10-03_DNN_ADAM_layer1x60_batch60_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir22= 'TrainedModels/models/2018-06-07_12-07_DNN_ADAM_layer3x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir23= 'TrainedModels/models/2018-06-11_14-38_DNN_ADAM_layer1x80_batch40_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir24= 'TrainedModels/models/2018-06-11_14-54_DNN_ADAM_layer1x80_batch40_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir25= 'TrainedModels/models/2018-06-11_16-42_DNN_ADAM_layer1x80_batch40_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir26= 'TrainedModels/models/2018-06-12_13-39_DNN_ADAM_layer1x100_batch50_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir27= 'TrainedModels/models/2018-06-12_16-48_DNN_ADAM_layer1x80_batch40_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir28= 'TrainedModels/models/2018-06-07_12-07_DNN_ADAM_layer3x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir29= 'TrainedModels/models/2018-06-11_14-35_DNN_ADAM_layer1x80_batch40_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir30= 'TrainedModels/models/2018-06-11_15-23_DNN_ADAM_layer3x128batch64_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir31= 'TrainedModels/models/2018-06-11_15-15_DNN_ADAM_layer3x128batch64_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    dirs = []
    #dirs.append(modelDir1)
    #dirs.append(modelDir2)
    #dirs.append(modelDir3)
    #dirs.append(modelDir4)
    #dirs.append(modelDir9)
    #dirs.append(modelDir16)
    #dirs.append(modelDir24)
    #dirs.append(modelDir27)
    #dirs.append(modelDir29)
    dirs.append(modelDir25)
    dirs2 = [x.replace('\n','') for x in open('notes/allModels2.txt').readlines()]
    dirs2errors = []
    for mdir in dirs:
        startPlot(mdir, save=True)
    #for mdir in dirs2:
        #try:
            #startPlot('TrainedModels/models/'+mdir, save=True)
        #except:
            #dirs2errors.append(mdir)
    print 'Following models could not be plotted:'
    print dirs2errors
    
if __name__== '__main__':
    main()
    
def runtimeSummary(t0):
  hour = t0 // 3600
  t0 %= 3600
  minutes = t0 // 60
  t0 %= 60
  seconds = t0

  print '-----Runtime Summary -----'
  print 'Job ran %d h:%d min:%d sec' % ( hour, minutes, seconds)
  print '--------------------------'

def getSequence(par):
  '''
    Creates a list of the trained sequences of an RNN
    from a string of parameters from the infofile

    Parameters
    ----------
    par : string that includes all the hyperparameters

    Returns
    -------
    seq : list of strings, which represents the trained sequences in an RNN e.g. ['jet','lep','bjet']
  '''

  match = [p for p in par if 'collection' in p][0]
  seq_str = match.replace('\" \'collection\': [','').replace(']\"','').split()[1]
  seq = seq_str.split('\'')
  seq.remove(seq[0])
  seq.remove(seq[-1])
  seq = filter(lambda x: x != ',', seq)

  return seq
