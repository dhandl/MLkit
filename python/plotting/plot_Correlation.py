import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


def plotCorrelation(s_train, b_train, s_test, b_test, nvar_raw, fileName='Test', save=False):
    print 'Plotting correlations between variables...'
    fileDir = "plots/"+fileName
    if save:
        if not os.path.exists("./plots/"):
            os.makedirs("./plots/")
            print("Creating folder plots")
            
    variables = [{'met': r'$E_{T}^{miss}$'}, {'mt': r'$m_{T}$'}, {'dphi_met_lep': r'$\Delta\Phi(l, E_{T}^{miss})$'}, {'amt2': '$am_{T2}$'}, {'n_jet': 'N jets'}, {'n_bjet': 'N bjets'}, {'jet_pt': r'$p_{T}^{jet}$'}, {'ht': r'$h_{T}$'}, {'dphi_jet0_ptmiss': r'$\Delta\Phi(jet0, p_{T}^{miss})$'}, {'dphi_jet1_ptmiss': r'$\Delta\Phi(jet1, p_{T}^{miss})$'}, {'lep_pt[0]': r'$p_{T}^{lep}$'}, {'m_bl': r'$m_{b,l}$'}, {'jet_pt[0]': r'$p_{T}^{jet0}$'}, {'jet_pt[1]': r'$p_{T}^{jet1}$'}, {'jet_pt[2]': r'$p_{T}^{jet2}$'}, {'jet_pt[3]': r'$p_{T}^{jet3}$'}, {'met_sig': r'$E_{T}^{miss}-sig$'}, {'ht_sig': r'$h_{T}-sig$'}, {'dphi_b_lep_max': r'$\max(\Delta\Phi(b, l))$'}, {'dphi_b_ptmiss_max': r'$\max(\Delta\Phi(b, p_{T}^{miss}))$'}, {'met_proj_lep': r'$E_{T}^{miss}-on-l$'}, {'dr_bjet_lep': r'$\Delta R(b,l)$'}, {'bjet_pt': r'$p_{T}^{bjet0}$'}, {'mT_blMET': r'$m_{T}^{blMET}$'}, {'bjet_pt[0]':r'$p_{T}^{bjet0}$'}]
    
    novars = []
    for v in variables:
        for x,y in v.iteritems():
            try:
                nvar_raw[nvar_raw.index(x)]=y
            except ValueError:
                novars.append(x)
                
    nvar = nvar_raw
    print 'Not in list for training: ' + ', '.join(novars)

    # Signal Training
    fig_x=10
    fig_y=10
    corr_strain, ax = plt.subplots(figsize=(fig_x,fig_y))
    df = pd.DataFrame(data=s_train, columns = nvar)
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', vmin=-1., vmax=1., square=True, fmt=".2f", annot=True)
    plt.title("Correlation Signal Training") 
    plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=True)
    if save:
        plt.savefig(fileDir+"_CorrelationMatrix_SignalTraining.pdf", bbox_inches='tight')
        plt.savefig(fileDir+"_CorrelationMatrix_SignalTraining.png", bbox_inches='tight')
        plt.close()
    # Background Training
    corr_btrain, ax = plt.subplots(figsize=(fig_x,fig_y))
    df = pd.DataFrame(data=b_train, columns = nvar)
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', vmin=-1., vmax=1., square=True, fmt=".2f", annot=True)
    plt.title("Correlation Background Training")
    plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=True)
    if save:
        plt.savefig(fileDir+"_CorrelationMatrix_BackgroundTraining.pdf", bbox_inches='tight')
        plt.savefig(fileDir+"_CorrelationMatrix_BackgroundTraining.png", bbox_inches='tight')
        plt.close()
    # Signal Training
    corr_stest, ax = plt.subplots(figsize=(fig_x,fig_y))
    df = pd.DataFrame(data=s_test, columns = nvar)
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', vmin=-1., vmax=1., square=True, fmt=".2f", annot=True)
    plt.title("Correlation Signal Test")
    plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=True)
    if save:
        plt.savefig(fileDir+"_CorrelationMatrix_SignalTest.pdf", bbox_inches='tight')
        plt.savefig(fileDir+"_CorrelationMatrix_SignalTest.png", bbox_inches='tight')
        plt.close()
    # Background Training
    corr_btest, ax = plt.subplots(figsize=(fig_x,fig_y))
    df = pd.DataFrame(data=b_test, columns = nvar)
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', vmin=-1., vmax=1., square=True, fmt=".2f", annot=True)
    plt.title("Correlation Background Test")
    plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=True)
    if save:
        plt.savefig(fileDir+"_CorrelationMatrix_BackgroundTest.pdf", bbox_inches='tight')
        plt.savefig(fileDir+"_CorrelationMatrix_BackgroundTest.png", bbox_inches='tight')
        plt.close()