import plot_piechart as pltp

from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from prepareTraining import prepareTraining

def evaluate_signalPoint(modelDir, signalPoints, save=False):
    
    '''
    Takes one point after the other from signalGrid and plots pie charts for predicted and true signal/backgrounds
    
    - modelDir: Directory of model
    - signalPoints: Signal points to look at, type=list
    '''
    print '-----NOT WORKING YET!-----'
    
    inputDir = '/project/etp5/dhandl/samples/SUSY/Stop1L/hdf5/cut_mt30_met60_preselection/'
    
    print 'Reading informations from ' , modelDir.replace(".h5","_infofile.txt") 
    
    infofile = open(modelDir.replace(".h5","_infofile.txt"))
    infos = infofile.readlines()
    
    variables=infos[4].replace('Used variables for training: ','').replace('\n','').split()
    preselection_raw=infos[6].replace('Used preselection: ', '').replace('; \n', '').split(';')
    weights=infos[5].replace('Used weights: ', '').replace('\n','').split()
    lumi=float(infos[7].replace('Used Lumi: ','').replace('\n',''))
    weight=infos[5].replace('Used weights: ','').replace('\n','').split()
    trainsize=float(infos[10].replace('Used trainsize/testsize: ','').replace('\n','').split('/')[0])
    testsize=float(infos[10].replace('Used trainsize/testsize: ','').replace('\n','').split('/')[1])
    reproduce=infos[11].replace('Used reproduce: ','').replace('\n','')
    multiclass=infos[2].replace('Used multiclass: ','').replace('\n','')
    
    background_list=infos[9].replace('Used background files: ','').replace('; \n','').replace(' ','').split(';')
    background = []
    for bkg in background_list:
        bkgdict = {'name':bkg, 'path':inputDir+ bkg + '/'}
        background.append(bkgdict)
    
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
       
    print 'Loading model...'   
    
    model = load_model(modelDir)
    model.load_weights(modelDir.replace(".h5" , "_weights.h5").replace("models" , "weights"))
    
    print 'Start plotting...'

    for signalPoint in signalPoints:
    
        signal = [{'name':signalPoint, 'path':inputDir+signalPoint + '/'}]
        
        filename = modelDir.replace("TrainedModels/models/","").replace(".h5","") + '_' + signalPoint
        dataset= 'TrainedModels/datasets/' + infos[3].replace('Used dataset: ','').replace('\n','') + '_' + signalPoint + '_test.h5'
        
        X_train, X_test, y_train, y_test, w_train, w_test = prepareTraining(signal, background, preselection, variables, weight, dataset, lumi, trainsize, testsize, reproduce, multiclass=multiclass)
        
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        y_predict_train = model.predict(X_train_scaled)
        y_predict_test = model.predict(X_test_scaled)
        
        pltp.plot_pie_chart(y_test, y_predict_test, fileName=filename, save=save)
    
    
    
    
    