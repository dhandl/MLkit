import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as mpatches
import AtlasStyle_mpl

#Other
import prepareTraining as pT
from collections import namedtuple
Sample = namedtuple('Sample', 'name dataframe')
from sklearn.preprocessing import StandardScaler

#def plot_classification(y_true, y_predict, save=False, fileName="Class_test"):
    #y_predict_class = np.argmax(y_predict, axis=1)
    ##plt.hist(y_predict_class[y_true==0], label=r'Signal', histtype='step')
    ##plt.hist(y_predict_class[y_true==1], label=r'$t\overline{t}$', histtype='step')
    ##plt.hist(y_predict_class[y_true==2], label=r'Single Top', histtype='step')
    ##plt.hist(y_predict_class[y_true==3], label=r'$W$ + Jets', histtype='step')
    
    
    #labels = [r'Signal',r'$t\overline{t}$',r'Single Top','$W$ + jets']
    ##hists = [y_predict_class[y_true==0],y_predict_class[y_true==1],y_predict_class[y_true==2],y_predict_class[y_true==3]] Predicted Class on x-axis
    #hists = [y_true[y_predict_class==0],y_true[y_predict_class==1],y_true[y_predict_class==2],y_true[y_predict_class==3]]
    
    #plt.hist(hists, bins=4, stacked=True,histtype='stepfilled', label=labels, density=True)
    
    #plt.xlabel('True Class')
    ##plt.yscale('log')
    #plt.legend(loc='best')
    #plt.xticks(np.arange(4),(r'Signal',r'$t\overline{t}$',r'Single Top','$W$ + jets'))
    #plt.title('Classification')
    
    ##print('Signal: ', y_predict_class[y_true==0].shape[0])
    ##print('tt: ', y_predict_class[y_true==1].shape[0])
    ##print('Single Top: ', y_predict_class[y_true==2].shape[0])
    ##print('W + jets: ', y_predict_class[y_true==3].shape[0])
    
    #print('Identified as:')
    #print('Signal: ', y_true[y_predict_class==0].shape[0])
    #print('tt: ', y_true[y_predict_class==1].shape[0])
    #print('Single Top: ', y_true[y_predict_class==2].shape[0])
    #print('W + jets: ', y_true[y_predict_class==3].shape[0])
    
    
    #if save:
        #plt.savefig("plots/"+fileName+".pdf")
        #plt.savefig("plots/"+fileName+".png")
        #plt.close()
    
def plot_classification(y_true, y_predict, weights, fileName="Test", save=False, weighted=False, train=False, sample=None, addStr=''):
    print('Plotting the classification for true labels...')
    if weighted:
        addStr+='_weighted'
    if train:
        addStr += '_train'
    if train and weighted:
        print 'For weighted events, whole dataset has to be used'
        return 0
    y_predict_class = np.argmax(y_predict, axis=1)
    classes = [0,1,2,3] #Different classes
    assignal = []
    astt = []
    assinglet = []
    asWjets = []
    
    explain_patch = mpatches.Patch(color='None', label="predicted label")
    
    if weighted:
        for i in range(0,4):
            assignal.append(np.sum(weights[np.logical_and(y_true==i, y_predict_class==0)]))
            astt.append(np.sum(weights[np.logical_and(y_true==i, y_predict_class==1)]))
            assinglet.append(np.sum(weights[np.logical_and(y_true==i, y_predict_class==2)]))
            asWjets.append(np.sum(weights[np.logical_and(y_true==i, y_predict_class==3)]))
    else:
        for i in range(0,4):
            n = float(y_predict_class[y_true==i].shape[0])

            u, counts = np.unique(y_predict_class[y_true==i], return_counts=True)
            
            #print(u.tolist())
            #print(counts.tolist())

            try:
                assignal.append(counts[u.tolist().index(0)]/n)
            except ValueError:
                assignal.append(0)
            try:
                astt.append(counts[u.tolist().index(1)]/n)
            except ValueError:
                astt.append(0)
            try:
                assinglet.append(counts[u.tolist().index(2)]/n)
            except ValueError:
                assinglet.append(0)
            try:
                asWjets.append(counts[u.tolist().index(3)]/n)
            except ValueError:
                asWjets.append(0)    
    
            
    width=1.
    
    bar0 = plt.bar(classes, assignal, width, label=r'Signal', color='r')
    bar1 = plt.bar(classes, astt, width, bottom=assignal, label=r'$t\overline{t}$', color='b')
    bar2 = plt.bar(classes, assinglet, width, bottom=np.array(astt)+np.array(assignal), label=r'Single Top', color='g')
    bar3 = plt.bar(classes, asWjets, width, bottom=np.array(assinglet)+np.array(astt)+np.array(assignal), label='$W$ + jets', color='orange')
    
    plt.xlabel('true label')
    #plt.legend(loc='best',handles=[explain_patch, bar0, bar1, bar2, bar3])
    plt.xticks(np.arange(4),(r'Signal',r'$t\overline{t}$',r'Single Top','$W$ + jets'))
    plt.title('Classification')
    
    if weighted:
        plt.ylim(0,max([assignal[i]+astt[i]+assinglet[i]+asWjets[i] for i in range(0,4)])*(1+0.33))
    
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    if sample is not None:
        sample_patch1 = mpatches.Patch(color='None', label=sample[0])
        sample_patch2 = mpatches.Patch(color='None', label=sample[1])
        plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5),handles=[explain_patch, bar0, bar1, bar2, bar3, sample_patch1, sample_patch2])
    else:
        plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5),handles=[explain_patch, bar0, bar1, bar2, bar3])
    
    if weighted:
        ax1=plt.gca()
        AtlasStyle_mpl.ATLASLabel(ax1, 0.02, 0.9, 'Work in progress')
        AtlasStyle_mpl.LumiLabel(ax1, 0.02, 0.8, lumi=140)
    
    #plt.gca().set_ylim([0,1.2])
    
    if save:
        if not os.path.exists("./plots/"):
            os.makedirs("./plots/")
            print("Creating folder plots")
        plt.savefig("plots/"+fileName+"_Classification"+addStr+".pdf")
        plt.savefig("plots/"+fileName+"_Classification"+addStr+".png")
        plt.close()
        
        
def plot_classification_datapoint(SignalList, model, preselection, nvar, weight, lumi, save=False, fileName='Test', multiclass=True):
    
    '''
    Evaluate the classification on certain datapoints. sigList is supposed to be in a form like in config/samples.py
    '''
    
    print '----- Plotting the Classification for different datapoints-----'
    
        
    print 'Using preselection', preselection
    
    #met_cut = False
    #for pre in preselection:
        #if pre['name'] == 'met':
            #met_cut = True
            #met_threshold = pre['threshold']
            #met_cut_addStr = 'met' + str(int(pre['threshold']*0.001))
    
    #if not met_cut:
        #print 'Using no met-preselection!'
    
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
      
    print 'Background shape', bkg.shape
      
    #Evaluating on signal for each set of points
    print 'Evaluating on signal sets...'
    
    for sigList in SignalList:
        Signal = []
        addStr = '_stop_bWN_'
        name=False
        title=''
        for s in sigList:
            if not name:
                addStr += s['name'].replace('stop_bWN_', '')
                name=True
            else:
                addStr += s['name'].replace(s['name'][:12], '')
            
            print 'Loading signal {} from {}...'.format(s['name'], s['path'])
            Signal.append(Sample(s['name'], pT.loadDataFrame(s['path'], preselection, nvar, weight, lumi)))
        
        title=addStr[1:17].replace('_', ' ')
        
        mstop = int(addStr[10:13])
        mneutralino = int(addStr[14:17])
        sample = [r'$m_{\tilde{t}}$=%i GeV' %mstop, r'$m_{\chi}$=%i GeV' %mneutralino]
    
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
        
        #if not met_cut:
            #addStr += '_no_met_cut'
            
        #print 'True classes:', y.shape, 'Predicted classes:', y_predict.shape
            
        #sig_predicted = deepcopy(y_predict)[y==0]
        #bkg_predicted = deepcopy(y_predict)[y!=0]
        #bkg1_predicted= deepcopy(y_predict)[y==1]
        #bkg2_predicted= deepcopy(y_predict)[y==2]
        #bkg3_predicted= deepcopy(y_predict)[y==3]
        #bkg1_w = deepcopy(w)[y==1]
        #bkg2_w = deepcopy(w)[y==2]
        #bkg3_w = deepcopy(w)[y==3]
        
        variables = nvar
        
        plot_classification(y, y_predict, w, save=save, fileName=fileName, weighted=True, sample=sample, addStr=addStr)
        plot_classification(y[X[:,variables.index('met')]>=250e3], y_predict[X[:,variables.index('met')]>=250e3], w[X[:,variables.index('met')]>=250e3], save=save, fileName=fileName, weighted=True, sample=sample, addStr=addStr+'_met250')