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
import plot_output_score2D
import plot_piechart
import evaluate_signalGrid

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
    
    print("Loading dataset...")
    
    infofile = open(modelDir.replace(".h5","_infofile.txt"))
    datasetDir = "TrainedModels/datasets/" + infofile.readlines()[3].replace("Used dataset: ", "").replace("\n","") + ".h5"
    
    dataset = h5py.File(datasetDir)
    
    filenames = modelDir.replace("TrainedModels/models/","").replace(".h5","")
    
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
    
    outputScore = y_predict_test[:,0]
    
    #Do various plots
    
    plot_TrainTest_score.plot_TrainTest_score(sig_predicted_train[:,0], sig_predicted_test[:,0], sig_w_train, sig_w_test, bkg_predicted_train[:,0], bkg_predicted_test[:,0], bkg_w_train, bkg_w_test, binning, normed=1,save=save,fileName=filenames)
    
    plt.figure()
    plot_ConfusionMatrix.plot_confusion_matrix(y_test, y_predict_test, filename=filenames, save=save, isTrain=False)
    
    plt.figure()
    plot_ConfusionMatrix.plot_confusion_matrix(y_train, y_predict_train, filename=filenames, save=save, isTrain=True)
    
    plt.figure()
    plot_Classification.plot_classification(y_test, y_predict_test, fileName=filenames, save=save)
    
    plt.figure()
    plot_learning_curve.learning_curve_for_keras(pickleDir, save=save, filename=filenames)
    
    plt.figure()
    plot_Classification2.plot_classification_2(y_test, y_predict_test, fileName=filenames, save=save)
    
    plt.figure()
    plot_piechart.plot_pie_chart(y_test, y_predict_test, fileName=filenames, save=save)
    
    plot_output_score.plot_output_score(sig_predicted_test[:,0], sig_w_test, bkg_predicted_test[:,0], bkg_w_test, binning, save=save, fileName=filenames)
    
    plt.figure()
    plot_output_score2D.plot_output_score2D(outputScore, X_test, r'$E_T^{miss}$', r'$m_T$', save=save, fileName=filenames)
    
    evaluate_signalGrid.evaluate_signalGrid(modelDir, save=save, fileName=filenames)
    
    # end timer and print time
    t.stop()
    t0 = t.elapsed
    t.reset()
    #runtimeSummary(t0)
    
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
    modelDir = 'TrainedModels/models/2018-05-18_15-33_DNN_ADAM_layer3x128_batch128_NormalInitializer_dropout0p5_l2-0p01_multiclass.h5'
    startPlot(modelDir, binning=[50,0,1.], save=False)
    
#if __name__== '__main__':
    #main()
    
def runtimeSummary(t0):
  hour = t0 // 3600
  t0 %= 3600
  minutes = t0 // 60
  t0 %= 60
  seconds = t0

  print '-----Runtime Summary -----'
  print 'Job ran %d h:%d min:%d sec' % ( hour, minutes, seconds)
  print '--------------------------'