#import os
#
#import h5py
#
#from keras.models import load_model
#from sklearn.preprocessing import StandardScaler
#
#import matplotlib.pyplot as plt
#
#def parse_options():
# import argparse
#
# workdir = os.getenv('WorkDir')
# output = os.path.join(workdir,'plots')
#
# parser = argparse.ArgumentParser()
# parser.add_argument('-m', '--model', help="Name of the model to plot" )
# parser.add_argument('-d', '--dataset', help="Name of the dataset to use")
#
# opts = parser.parse_args()
#
# return opts
#
#def main():
# datasetDir = opts.dataset
# modelDir = opts.model
# weightsDir = opts.model.replace(".h5" , "_weights.h5")
#
# dataset = h5py.File(datasetDir)
# model = load_model(modelDir)
# model.load_weights(weightsDir)
#
# X_train = dataset["X_train"][:]
# X_test = dataset["X_test"][:]
#
# scaler = StandardScaler()
#
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# y_predict = model.predict(X1test_scaled)
#
# plt.hist(y_predict)
# plt.show()

##

#def select_samples(path):
    #df = pd.read_hdf(path)
    #l = df.shape[0]
    #samples = np.zeros(l)
    ##Debug Variables
    #countkey = 0
    #countindex = 0
    
    #for var in variables.nvar:
        #try:
            #samples = np.c_[samples,df[var]]
        #except KeyError:
            #countkey += 1
            #var = var.replace("[0]","")
            #samples = np.c_[samples,np.zeros(l)]
            #for i in range(0,l-1):
                #try:
                    #samples[i,samples.shape[1]-1] = df[var][i][0]
                #except IndexError:
                    #countindex += 1
    #samples = samples[:,1:]
    #print("KeyErrors:" ,countkey, "IndexErrors: ", countindex)
    #return samples
    
import pandas as pd
import variables
import numpy as np

def select_variables(path):
    """
    Selects certain variables from sample stored in path
    """
    df = pd.read_hdf(path)
    l = df.shape[0]
    samples = np.zeros(l)
    for var in variables.nvar:
        try:
            samples = np.c_[samples,df[var]]
        except KeyError:
            var = var.replace("[0]","")
            samples = np.c_[samples,df[var].str[0]]
    samples = samples[:,1:]
    return samples