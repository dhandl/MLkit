import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

def plotCorrelation(X, y_predict, y_class, nvar_raw, fileName='Test', save=False, plotEPD=True):
    print 'Plotting correlations between variables...'
    fileDir = "plots/"+fileName
    if save:
        if not os.path.exists("./plots/"):
            os.makedirs("./plots/")
            print("Creating folder plots")
            
    variables = [{'met': r'$E_{T}^{miss}$'}, {'mt': r'$m_{T}$'}, {'dphi_met_lep': r'$\Delta\Phi(l, E_{T}^{miss})$'}, {'amt2': '$am_{T2}$'}, {'n_jet': 'N jets'}, {'n_bjet': 'N bjets'}, {'jet_pt': r'$p_{T}^{jet}$'}, {'ht': r'$h_{T}$'}, {'dphi_jet0_ptmiss': r'$\Delta\Phi(jet0, p_{T}^{miss})$'}, {'dphi_jet1_ptmiss': r'$\Delta\Phi(jet1, p_{T}^{miss})$'}, {'lep_pt[0]': r'$p_{T}^{lep}$'}, {'m_bl': r'$m_{b,l}$'}, {'jet_pt[0]': r'$p_{T}^{jet0}$'}, {'jet_pt[1]': r'$p_{T}^{jet1}$'}, {'jet_pt[2]': r'$p_{T}^{jet2}$'}, {'jet_pt[3]': r'$p_{T}^{jet3}$'}, {'met_sig': r'$E_{T}^{miss, sig}$'}, {'ht_sig': r'$h_{T}^{sig}$'}, {'dphi_b_lep_max': r'$\max(\Delta\Phi(b, l))$'}, {'dphi_b_ptmiss_max': r'$\max(\Delta\Phi(b, p_{T}^{miss}))$'}, {'met_proj_lep': r'$E_{T,l}^{miss}$'}, {'dr_bjet_lep': r'$\Delta R(b,l)$'}, {'bjet_pt': r'$p_{T}^{bjet0}$'}, {'mT_blMET': r'$m_{T}^{blMET}$'}, {'bjet_pt[0]':r'$p_{T}^{bjet0}$'},{'jet_eta[0]':r'$\eta^{jet0}$'},{'jet_eta[1]':r'$\eta^{jet1}$'},{'jet_eta[2]':r'$\eta^{jet2}$'},{'jet_eta[3]':r'$\eta^{jet3}$'},{'jet_phi[0]':r'$\Phi^{jet0}$'},{'jet_phi[1]':r'$\Phi^{jet1}$'},{'jet_phi[2]':r'$\Phi^{jet2}$'},{'jet_phi[3]':r'$\Phi^{jet3}$'},{'met_phi':r'$\Phi(E_{T}^{miss})$'},{'lep_eta[0]':r'$\eta^{lep0}$'},{'lep_phi[0]':r'$\Phi^{lep0}$'},{'jet_e[0]':r'$E^{jet0}$'},{'jet_e[1]':r'$E^{jet1}$'},{'jet_e[2]':r'$E^{jet2}$'},{'jet_e[3]':r'$E^{jet3}$'},{'lep_e[0]':r'$E^{lep0}$'},{'jet_mv2c10[0]':r'$b_{tag}^{jet0}$'},{'jet_mv2c10[1]':r'$b_{tag}^{jet1}$'},{'jet_mv2c10[2]':r'$b_{tag}^{jet2}$'},{'jet_mv2c10[3]':r'$b_{tag}^{jet3}$'}]
    
    novars = []
    for v in variables:
        for x,y in v.iteritems():
            try:
                nvar_raw[nvar_raw.index(x)]=y
            except ValueError:
                novars.append(x)
                
    nvar = nvar_raw[:]
    
    addStr=''
    if plotEPD:
        nvar.append('EPD') #Output Score or event probability discriminator
    else:
        addStr = '_noEPD'
        
    
    s = X[y_class==0]
    b = X[y_class!=0]
    b1 = X[y_class==1]
    b2 = X[y_class==2]
    b3 = X[y_class==3]
    output_score = y_predict[:,0]
    out_s = output_score[y_class==0]
    out_b = output_score[y_class!=0]
    out_b1 = output_score[y_class==1]
    out_b2 = output_score[y_class==2]
    out_b3 = output_score[y_class==3]
    
    if plotEPD:
        s = np.concatenate((s,np.transpose(out_s.reshape(1,len(out_s)))), axis=1)
        b = np.concatenate((b,np.transpose(out_b.reshape(1,len(out_b)))), axis=1)
        b1 = np.concatenate((b1,np.transpose(out_b1.reshape(1,len(out_b1)))), axis=1)
        b2 = np.concatenate((b2,np.transpose(out_b2.reshape(1,len(out_b2)))), axis=1)
        b3 = np.concatenate((b3,np.transpose(out_b3.reshape(1,len(out_b3)))), axis=1)
    
    print 'Not in list for training: ' + ', '.join(novars)

    if len(nvar)<=20:
        fig_x=10
        fig_y=10
    else:
        fig_x=20
        fig_y=20
        
    font_scale=1.2
    fs=12
         
    #Signal
    corr_strain, ax = plt.subplots(figsize=(fig_x,fig_y))
    df = pd.DataFrame(data=s, columns = nvar)
    corr = df.corr()
    sns.set(font_scale=font_scale)
    sns.heatmap(corr, cmap='coolwarm', vmin=-1., vmax=1., square=True, fmt=".2f", annot=True)
    plt.title("Correlation Signal",fontsize=fs) 
    plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=True)
    if save:
        plt.savefig(fileDir+"_CorrelationMatrix"+addStr+"_Signal.pdf", bbox_inches='tight')
        plt.savefig(fileDir+"_CorrelationMatrix"+addStr+"_Signal.png", bbox_inches='tight')
        plt.close()
        
    #For no reasons the first plot differs in pixel size compared to the others
    #As this looks rather stupid when directly comparing, the first (wrong sized) plot is overwritten with the same plot, but correct size
    #Probably this is the least elegant way to do this, but it works
        
    #Signal
    corr_strain, ax = plt.subplots(figsize=(fig_x,fig_y))
    df = pd.DataFrame(data=s, columns = nvar)
    corr = df.corr()
    sns.set(font_scale=font_scale)
    sns.heatmap(corr, cmap='coolwarm', vmin=-1., vmax=1., square=True, fmt=".2f", annot=True)
    plt.title("Correlation Signal",fontsize=fs) 
    plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=True)
    if save:
        plt.savefig(fileDir+"_CorrelationMatrix"+addStr+"_Signal.pdf", bbox_inches='tight')
        plt.savefig(fileDir+"_CorrelationMatrix"+addStr+"_Signal.png", bbox_inches='tight')
        plt.close()
    #All Background
    corr_btrain, ax = plt.subplots(figsize=(fig_x,fig_y))
    df = pd.DataFrame(data=b, columns = nvar)
    corr = df.corr()
    sns.set(font_scale=font_scale)
    sns.heatmap(corr, cmap='coolwarm', vmin=-1., vmax=1., square=True, fmt=".2f", annot=True)
    plt.title("Correlation Background",fontsize=fs)
    plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=True)
    if save:
        plt.savefig(fileDir+"_CorrelationMatrix"+addStr+"_Background_all.pdf", bbox_inches='tight')
        plt.savefig(fileDir+"_CorrelationMatrix"+addStr+"_Background_all.png", bbox_inches='tight')
        plt.close()
    # ttbar
    corr_stest, ax = plt.subplots(figsize=(fig_x,fig_y))
    df = pd.DataFrame(data=b1, columns = nvar)
    corr = df.corr()
    sns.set(font_scale=font_scale)
    sns.heatmap(corr, cmap='coolwarm', vmin=-1., vmax=1., square=True, fmt=".2f", annot=True)
    plt.title(r"Correlation $t\overline{t}$",fontsize=fs)
    plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=True)
    if save:
        plt.savefig(fileDir+"_CorrelationMatrix"+addStr+"_Background_ttbar.pdf", bbox_inches='tight')
        plt.savefig(fileDir+"_CorrelationMatrix"+addStr+"_Background_ttbar.png", bbox_inches='tight')
        plt.close()
    # single top
    corr_btest, ax = plt.subplots(figsize=(fig_x,fig_y))
    df = pd.DataFrame(data=b2, columns = nvar)
    corr = df.corr()
    sns.set(font_scale=font_scale)
    sns.heatmap(corr, cmap='coolwarm', vmin=-1., vmax=1., square=True, fmt=".2f", annot=True)
    plt.title("Correlation single top",fontsize=fs)
    plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=True)
    if save:
        plt.savefig(fileDir+"_CorrelationMatrix"+addStr+"_Background_singletop.pdf", bbox_inches='tight')
        plt.savefig(fileDir+"_CorrelationMatrix"+addStr+"_Background_singletop.png", bbox_inches='tight')
        plt.close()        
    # w + jets
    corr_btest, ax = plt.subplots(figsize=(fig_x,fig_y))
    df = pd.DataFrame(data=b3, columns = nvar)
    corr = df.corr()
    sns.set(font_scale=font_scale)
    sns.heatmap(corr, cmap='coolwarm', vmin=-1., vmax=1., square=True, fmt=".2f", annot=True)
    plt.title(r"Correlation $W$ + jets")
    plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=True)
    if save:
        plt.savefig(fileDir+"_CorrelationMatrix"+addStr+"_Background_wjets.pdf", bbox_inches='tight')
        plt.savefig(fileDir+"_CorrelationMatrix"+addStr+"_Background_wjets.png", bbox_inches='tight')
        plt.close()


#def plotCorrelation(s_train, b_train, s_test, b_test, out_s_train, out_s_test, out_b_train, out_b_test, nvar_raw, fileName='Test', save=False):
    #print 'Plotting correlations between variables...'
    #fileDir = "plots/"+fileName
    #if save:
        #if not os.path.exists("./plots/"):
            #os.makedirs("./plots/")
            #print("Creating folder plots")
            
    #variables = [{'met': r'$E_{T}^{miss}$'}, {'mt': r'$m_{T}$'}, {'dphi_met_lep': r'$\Delta\Phi(l, E_{T}^{miss})$'}, {'amt2': '$am_{T2}$'}, {'n_jet': 'N jets'}, {'n_bjet': 'N bjets'}, {'jet_pt': r'$p_{T}^{jet}$'}, {'ht': r'$h_{T}$'}, {'dphi_jet0_ptmiss': r'$\Delta\Phi(jet0, p_{T}^{miss})$'}, {'dphi_jet1_ptmiss': r'$\Delta\Phi(jet1, p_{T}^{miss})$'}, {'lep_pt[0]': r'$p_{T}^{lep}$'}, {'m_bl': r'$m_{b,l}$'}, {'jet_pt[0]': r'$p_{T}^{jet0}$'}, {'jet_pt[1]': r'$p_{T}^{jet1}$'}, {'jet_pt[2]': r'$p_{T}^{jet2}$'}, {'jet_pt[3]': r'$p_{T}^{jet3}$'}, {'met_sig': r'$E_{T}^{miss, sig}$'}, {'ht_sig': r'$h_{T}^{sig}'}, {'dphi_b_lep_max': r'$\max(\Delta\Phi(b, l))$'}, {'dphi_b_ptmiss_max': r'$\max(\Delta\Phi(b, p_{T}^{miss}))$'}, {'met_proj_lep': r'$E_{T,l}^{miss}$'}, {'dr_bjet_lep': r'$\Delta R(b,l)$'}, {'bjet_pt': r'$p_{T}^{bjet0}$'}, {'mT_blMET': r'$m_{T}^{blMET}$'}, {'bjet_pt[0]':r'$p_{T}^{bjet0}$'}]
    
    #novars = []
    #for v in variables:
        #for x,y in v.iteritems():
            #try:
                #nvar_raw[nvar_raw.index(x)]=y
            #except ValueError:
                #novars.append(x)
                
    #nvar = nvar_raw
    #nvar.append('output-score') #Output Score
    
    #s_train = np.concatenate((s_train,np.transpose(out_s_train.reshape(1,len(out_s_train)))), axis=1)
    #s_test = np.concatenate((s_test,np.transpose(out_s_test.reshape(1,len(out_s_test)))), axis=1)
    #b_train = np.concatenate((b_train,np.transpose(out_b_train.reshape(1,len(out_b_train)))), axis=1)
    #b_test = np.concatenate((b_test,np.transpose(out_b_test.reshape(1,len(out_b_test)))), axis=1)
    
    #print 'Not in list for training: ' + ', '.join(novars)

    ## Signal Training
    #fig_x=10
    #fig_y=10
    #corr_strain, ax = plt.subplots(figsize=(fig_x,fig_y))
    #df = pd.DataFrame(data=s_train, columns = nvar)
    #corr = df.corr()
    #sns.heatmap(corr, cmap='coolwarm', vmin=-1., vmax=1., square=True, fmt=".2f", annot=True)
    #plt.title("Correlation Signal Training") 
    #plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=True)
    #if save:
        #plt.savefig(fileDir+"_CorrelationMatrix_SignalTraining.pdf", bbox_inches='tight')
        #plt.savefig(fileDir+"_CorrelationMatrix_SignalTraining.png", bbox_inches='tight')
        #plt.close()
    ## Background Training
    #corr_btrain, ax = plt.subplots(figsize=(fig_x,fig_y))
    #df = pd.DataFrame(data=b_train, columns = nvar)
    #corr = df.corr()
    #sns.heatmap(corr, cmap='coolwarm', vmin=-1., vmax=1., square=True, fmt=".2f", annot=True)
    #plt.title("Correlation Background Training")
    #plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=True)
    #if save:
        #plt.savefig(fileDir+"_CorrelationMatrix_BackgroundTraining.pdf", bbox_inches='tight')
        #plt.savefig(fileDir+"_CorrelationMatrix_BackgroundTraining.png", bbox_inches='tight')
        #plt.close()
    ## Signal Training
    #corr_stest, ax = plt.subplots(figsize=(fig_x,fig_y))
    #df = pd.DataFrame(data=s_test, columns = nvar)
    #corr = df.corr()
    #sns.heatmap(corr, cmap='coolwarm', vmin=-1., vmax=1., square=True, fmt=".2f", annot=True)
    #plt.title("Correlation Signal Test")
    #plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=True)
    #if save:
        #plt.savefig(fileDir+"_CorrelationMatrix_SignalTest.pdf", bbox_inches='tight')
        #plt.savefig(fileDir+"_CorrelationMatrix_SignalTest.png", bbox_inches='tight')
        #plt.close()
    ## Background Training
    #corr_btest, ax = plt.subplots(figsize=(fig_x,fig_y))
    #df = pd.DataFrame(data=b_test, columns = nvar)
    #corr = df.corr()
    #sns.heatmap(corr, cmap='coolwarm', vmin=-1., vmax=1., square=True, fmt=".2f", annot=True)
    #plt.title("Correlation Background Test")
    #plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=True)
    #if save:
        #plt.savefig(fileDir+"_CorrelationMatrix_BackgroundTest.pdf", bbox_inches='tight')
        #plt.savefig(fileDir+"_CorrelationMatrix_BackgroundTest.png", bbox_inches='tight')
        #plt.close()