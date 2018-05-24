import sys
from sklearn.preprocessing import StandardScaler
import h5py
from keras.models import load_model
import matplotlib.pyplot as plt
import timer
from sklearn.externals import joblib
import pickle

sys.path.append("./python/plotting/")

import plot_TrainTest_score
import plot_ConfusionMatrix
import plot_Classification
import plot_Classification2
import plot_learning_curve
import plot_output_score
import plot_output_score_multiclass
import plot_output_score2D
import plot_piechart
import evaluate_signalGrid
import plot_ROCcurves

def startPlot(modelDir, binning=[50,0,1.], save=False):
    """
    Plot all important things
        
    - modelDir: Directory of model
    
    - binning = [bins, start, stop] default: [50,0,1.]
    
    - save: Save Files in ./plots/ (True/False)
    """
    t = timer.Timer()
    t.start()
    
    #Load models
    
    print 'Loading infos from infofile...'
    
    infofile = open(modelDir.replace(".h5","_infofile.txt"))
    infos = infofile.readlines()
    
    variables=infos[4].replace('Used variables for training: ','').replace('\n','').split()
    weights=infos[5].replace('Used weights: ', '').replace('\n','').split()
    lumi=float(infos[7].replace('Used Lumi: ','').replace('\n',''))
    
    preselection_raw=infos[6].replace('Used preselection: ', '').replace('; \n', '').split(';')
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
        
    print "Loading dataset..."
    datasetDir = "TrainedModels/datasets/" + infos[3].replace("Used dataset: ", "").replace("\n","") + ".h5"
    
    dataset = h5py.File(datasetDir)
    
    filenames = modelDir.replace("TrainedModels/models/","").replace(".h5","")
    
    #filenames='TEST_20180524'
    
    print "Using dataset from:", datasetDir
    
    print 'Reading files for specific confusion matrices...'
    
    Signal = []
    input = '/project/etp5/dhandl/samples/SUSY/Stop1L/hdf5/cut_mt30_met60_preselection/'
    
    point1 = [{'name':'stop_bWN_250_100', 'path':input+'stop_bWN_250_100/'},{'name':'stop_bWN_250_130', 'path':input+'stop_bWN_250_130/'},{'name':'stop_bWN_250_160', 'path':input+'stop_bWN_250_160/'}]
    point2 = [{'name':'stop_bWN_300_150', 'path':input+'stop_bWN_300_150/'},{'name':'stop_bWN_300_180', 'path':input+'stop_bWN_300_180/'},{'name':'stop_bWN_300_210', 'path':input+'stop_bWN_300_210/'}]
    point3 = [{'name':'stop_bWN_350_200', 'path':input+'stop_bWN_350_200/'},{'name':'stop_bWN_350_230', 'path':input+'stop_bWN_350_230/'},{'name':'stop_bWN_350_260', 'path':input+'stop_bWN_350_260/'}]
    point4 = [{'name':'stop_bWN_400_250', 'path':input+'stop_bWN_400_250/'},{'name':'stop_bWN_400_280', 'path':input+'stop_bWN_400_280/'},{'name':'stop_bWN_400_310', 'path':input+'stop_bWN_400_310/'}]
    point5 = [{'name':'stop_bWN_450_300', 'path':input+'stop_bWN_450_300/'},{'name':'stop_bWN_450_330', 'path':input+'stop_bWN_450_330/'},{'name':'stop_bWN_450_360', 'path':input+'stop_bWN_450_360/'}]
    point6 = [{'name':'stop_bWN_500_350', 'path':input+'stop_bWN_500_350/'},{'name':'stop_bWN_500_380', 'path':input+'stop_bWN_500_380/'}]
    point7 = [{'name':'stop_bWN_550_400', 'path':input+'stop_bWN_550_400/'},{'name':'stop_bWN_550_430', 'path':input+'stop_bWN_550_430/'},{'name':'stop_bWN_550_460', 'path':input+'stop_bWN_550_460/'}]
    point8 = [{'name':'stop_bWN_600_450', 'path':input+'stop_bWN_600_450/'},{'name':'stop_bWN_600_480', 'path':input+'stop_bWN_600_480/'},{'name':'stop_bWN_600_510', 'path':input+'stop_bWN_600_510/'}]
    point9 = [{'name':'stop_bWN_650_500', 'path':input+'stop_bWN_650_500/'},{'name':'stop_bWN_650_530', 'path':input+'stop_bWN_650_530/'},{'name':'stop_bWN_650_560', 'path':input+'stop_bWN_650_560/'}]
    
    Signal.append(point1)
    Signal.append(point2)
    Signal.append(point3)
    Signal.append(point4)
    Signal.append(point5)
    Signal.append(point6)
    Signal.append(point7)
    Signal.append(point8)
    Signal.append(point9)
    
    print "Loading model..."
    
    try:
        pickleDir = modelDir.replace(".h5", "_history.pkl")
        model = load_model(modelDir)
        model.load_weights(modelDir.replace(".h5" , "_weights.h5").replace("models" , "weights"))
        print("Neuronal Network detected!")
        print("Scaling and reading values...")
    except IOError:
        model = joblib.load(modelDir)
        print("Boosted Decision Tree detected!")
        return 0
    
    #Get the data and scale it, if necessary
    print 'Loading data from dataset...'
    
    X_train = dataset["X_train"][:]
    X_test = dataset["X_test"][:]
    y_train= dataset["y_train"][:]
    y_test= dataset["y_test"][:]
       
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_predict_train = model.predict(X_train_scaled)
    y_predict_test = model.predict(X_test_scaled)
    
    sig_predicted_train = y_predict_train[y_train==0]
    sig_predicted_test = y_predict_test[y_test==0]
    sig_w_train = dataset["w_train"][y_train==0]
    sig_w_test = dataset["w_test"][y_test==0]
    bkg_predicted_train = y_predict_train[y_train!=0]
    bkg_predicted_test = y_predict_test[y_test!=0]
    bkg_w_train = dataset["w_train"][y_train!=0]
    bkg_w_test = dataset["w_test"][y_test!=0]
    
    bkg1_predicted_test = y_predict_test[y_test==1]
    bkg1_w_test = dataset["w_test"][y_test==1]
    bkg2_predicted_test = y_predict_test[y_test==2]
    bkg2_w_test = dataset["w_test"][y_test==2]
    bkg3_predicted_test = y_predict_test[y_test==3]
    bkg3_w_test = dataset["w_test"][y_test==3]
    
    outputScore = y_predict_test[:,0]
    
    #Do various plots
    print 'Start plotting...'
    
    plot_TrainTest_score.plot_TrainTest_score(sig_predicted_train[:,0], sig_predicted_test[:,0], sig_w_train, sig_w_test, bkg_predicted_train[:,0], bkg_predicted_test[:,0], bkg_w_train, bkg_w_test, binning, normed=1,save=save,fileName=filenames)
    
    plt.figure()
    plot_ConfusionMatrix.plot_confusion_matrix(y_test, y_predict_test, filename=filenames, save=save, isTrain=False)
    
    plt.figure()
    plot_ConfusionMatrix.plot_confusion_matrix(y_train, y_predict_train, filename=filenames, save=save, isTrain=True)
    
    print '----- Plotting the confusion matrices for different datapoints-----'
    plt.figure()
    plot_ConfusionMatrix.plot_confusion_matrix_datapoint(Signal, model, preselection, variables, weights, lumi, save=save, fileName=filenames)
    
    plt.figure()
    plot_Classification.plot_classification(y_test, y_predict_test, fileName=filenames, save=save)
    
    plt.figure()
    plot_learning_curve.learning_curve_for_keras(pickleDir, save=save, filename=filenames)
    
    plt.figure()
    plot_Classification2.plot_classification_2(y_test, y_predict_test, fileName=filenames, save=save)
    
    plt.figure()
    plot_piechart.plot_pie_chart(y_test, y_predict_test, fileName=filenames, save=save)
    
    plot_output_score.plot_output_score(sig_predicted_test[:,0], sig_w_test, bkg_predicted_test[:,0], bkg_w_test, binning, save=save, fileName=filenames)
    
    plot_output_score_multiclass.plot_output_score_multiclass(sig_predicted_test[:,0], sig_w_test, bkg1_predicted_test[:,0], bkg1_w_test, bkg2_predicted_test[:,0], bkg2_w_test, bkg3_predicted_test[:,0], bkg3_w_test, bkg_predicted_test[:,0], bkg_w_test, binning, save=save, fileName=filenames)
    
    plt.figure()
    plot_output_score2D.plot_output_score2D(outputScore, X_test, r'$E_T^{miss}$', r'$m_T$', save=save, fileName=filenames)
    
    plot_ROCcurves.plot_ROC(y_train, y_test, y_predict_train, y_predict_test, save=save, fileName=filenames)
    
    evaluate_signalGrid.evaluate_signalGrid(modelDir, save=save, fileName=filenames)
    
    # end timer and print time
    t.stop()
    t0 = t.elapsed
    t.reset()
    runtimeSummary(t0)
    
def startPlotDataset(modelDir, datasetDir, binning=[50,0,1.], save=False):
    """
    Plot all important things for specific dataset, which is not the one used for training
        
    - modelDir: Directory of model
    
    - datasetDir: Directory of dataset
    
    - binning = [bins, start, stop] default: [50,0,1.]
    
    - save: Save Files in ./plots/ (True/False)
    """
    t = timer.Timer()
    t.start()
    
    #Load models
    
    print("Loading dataset...")
    
    dataset = h5py.File(datasetDir)
    
    filenames = modelDir.replace("TrainedModels/models/","").replace(".h5","") + '_differentDataset'
    
    print("Using dataset from:", datasetDir)
    
    print("Loading model...")
    
    try:
        pickleDir = modelDir.replace(".h5", "_history.pkl")
        model = load_model(modelDir)
        model.load_weights(modelDir.replace(".h5" , "_weights.h5").replace("models" , "weights"))
        print("Neuronal Network detected!")
        print("Scaling and reading values...")
    except IOError:
        model = joblib.load(modelDir)
        print("Boosted Decision Tree detected!")
        return 0
    
    #Get the data and scale it, if necessary
    
    X = dataset["X"][:]
    y = dataset["y"][:]
    
    scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X)
    
    y_predict = model.predict(X_scaled)
    
    sig_predicted = y_predict[y==0]
    sig_w = dataset["w"][y==0]
    
    bkg_predicted= y_predict[y!=0]
    bkg_w = dataset["w"][y!=0]
    
    #Do various plots
    
    #plt.figure()
    #plot_ConfusionMatrix.plot_confusion_matrix(y, y_predict, filename=filenames, save=save, isTrain=False)
    
    #plt.figure()
    #plot_Classification.plot_classification(y, y_predict, fileName=filenames, save=save)
    
    #plt.figure()
    #plot_Classification2.plot_classification_2(y, y_predict, fileName=filenames, save=save)
    
    #plt.figure()
    #plot_piechart.plot_pie_chart(y, y_predict, fileName=filenames, save=save)
    
    #plot_output_score.plot_output_score(sig_predicted[:,0], sig_w, bkg_predicted[:,0], bkg_w, binning, save=save, fileName=filenames)
    
    evaluate_signalGrid.evaluate_signalGridCut(modelDir, save=save, fileName=filenames)
    
    # end timer and print time
    t.stop()
    t0 = t.elapsed
    t.reset()
    runtimeSummary(t0)
    
def main():
    print '---------- Warning: startPlot in main ----------'
    modelDir1 = 'TrainedModels/models/2018-05-18_15-33_DNN_ADAM_layer3x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir2 = 'TrainedModels/models/2018-05-18_15-04_DNN_ADAM_layer1x80_batch40_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir3 = 'TrainedModels/models/2018-05-18_15-12_DNN_ADAM_layer1x80_batch40_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir4 = 'TrainedModels/models/2018-05-18_15-26_DNN_ADAM_layer3x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    modelDir = 'TrainedModels/models/2018-05-17_10-44_DNN_ADAM_layer4x128_batch100_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    startPlot(modelDir2, binning=[50,0,1.], save=True)
    
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