import os

from separation import getSeparation
from getRatio import getRatio
from getSumW2 import getSumW2
# arrays
import numpy as np
import pandas as pd

# matplot lib for plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import plot_output_score_multiclass
import AtlasStyle_mpl

#Other
import prepareTraining as pT
from collections import namedtuple
from copy import deepcopy

Sample = namedtuple('Sample', 'name dataframe')

from sklearn.preprocessing import StandardScaler

def plot_output_score(sig_predicted, sig_w, bkg_predicted, bkg_w, binning, fileName='Test', normed=False, save=False, addStr='', ratio=True, log=False, sample=None):
  print('Plotting the binary output score...')
  fig = plt.figure(figsize=(8,6))
  if ratio:
    ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=3)
    ax1.set_xlabel('', fontsize=0.)
    ax1.set_xticklabels(())
  else: 
    ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=4)
  ax1.tick_params(direction='in')
  ax1.set_xlim((binning[1], binning[2]))
  ax1.xaxis.set_ticks_position('both')
  ax1.yaxis.set_ticks_position('both')

  s_hist, s_bins, s_patches = plt.hist(sig_predicted.ravel(), weights=sig_w, histtype='stepfilled', color='r', label='Signal', alpha=0.5, bins=binning[0], range=(binning[1], binning[2]), density=normed)
  b_hist, b_bins, b_patches = plt.hist(bkg_predicted.ravel(), weights=bkg_w, histtype='stepfilled', color='b', label='Background', alpha=0.5, bins=binning[0], range=(binning[1], binning[2]), density=normed)
  
  log_str = ''
  
  if log:
      plt.yscale('log', nonposy='clip')
      log_str = '_log'

  s_w = getSumW2(sig_predicted.ravel(), sig_w, binning)
  b_w = getSumW2(bkg_predicted.ravel(), bkg_w, binning)

  #sep = getSeparation(s_histTest, s_binsTest, b_histTest, b_binsTest)

  #print sep

  if normed:
    ax1.set_ylabel('a. u.', horizontalalignment='right', x=1.0)
  else:
    ax1.set_ylabel('Events', horizontalalignment='right', y=1.0)
  
  #ax1.set_ylim((0, s_hist.max()*(1+0.33)))
  
  if log:
      ax1.set_ylim((0, b_hist.max()*(30)))
  else:
      ax1.set_ylim((0, b_hist.max()*(1+0.33)))
      
  if sample is not None:
    sample_patch = mpatches.Patch(color='None', label=sample)
    leg = plt.legend(loc='best', frameon=False, handles=[s_patches[0], b_patches[0], sample_patch])
  else:
    leg = plt.legend(loc='best', frameon=False)
  p = leg.get_window_extent()
  #ax.annotate('KS Test S (B): %.3f (%.3f)'%(ks_sig, ks_bkg),(p.p0[0], p.p1[1]), (p.p0[0], p.p1[1]), xycoords='figure pixels', zorder=9)
  #ax1.text(0.65, 0.7, 'KS Test S (B): %.3f (%.3f)'%(ks_sig, ks_bkg), transform=ax1.transAxes)
  #ax1.text(0.65, 0.70, '$<S^2>$ = %.3f'%(sep), transform=ax1.transAxes)
  #ax.text(0.55, 0.7, 'KS p-value S (B): %.3f (%.3f)'%(ks_sig_p, ks_bkg_p), transform=ax.transAxes)
  
  AtlasStyle_mpl.ATLASLabel2(ax1, 0.02, 0.9, 'Work in progress')
  AtlasStyle_mpl.LumiLabel(ax1, 0.02, 0.8, lumi=140)
  if ratio:
    ax2 = plt.subplot2grid((4,4), (3,0), colspan=4, rowspan=1)
    r = getRatio(s_hist, s_bins, s_w, b_hist, b_bins, b_w, 'r')
    ax2.set_xlabel('EPD', horizontalalignment='right', x=1.0)
    ax2.set_ylabel('S/B')
    ax2.set_xlim((binning[1],binning[2]))
    ax2.set_ylim((-0.5,2.5))
    ax2.grid()
    ax2.tick_params(direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')

  ax1.set_xlabel('EPD', horizontalalignment='right', x=1.0)

  if save:
    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
        print('Creating folder plots')
    plt.savefig('plots/'+fileName+'_output_score' + addStr + log_str +'.pdf')
    plt.savefig('plots/'+fileName+'_output_score' + addStr + log_str +'.png')
    plt.close()
  return r, s_bins

def plot_output_score_datapoint(SignalList, model, preselection, nvar, weight, lumi, binning, save=False, fileName='Test', multiclass=True):
    
    '''
    Evaluate the output score on certain datapoints. sigList is supposed to be in a form like in config/samples.py
    '''
    
    print '----- Plotting the output score for different datapoints-----'
    
        
    print 'Using preselection', preselection
    
    met_cut = False
    for pre in preselection:
        if pre['name'] == 'met':
            met_cut = True
            met_threshold = pre['threshold']
            met_cut_addStr = 'met' + str(int(pre['threshold']*0.001))
    
    if not met_cut:
        print 'Using no met-preselection!'
    
    input='/project/etp5/dhandl/samples/SUSY/Stop1L/FullRun2/hdf5/cut_mt30_met60_preselection/'
    
    bkgList = [
    {'name':'powheg_ttbar', 'path':input+'mc16d_ttbar/'},
    {'name':'powheg_singletop', 'path':input+'mc16d_singletop/'},
    {'name':'sherpa22_Wjets', 'path':input+'mc16d_Wjets/'}
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
        sample = r'$m_{\tilde{t}}$=%i GeV, $m_{\chi}$=%i GeV'%(mstop, mneutralino)
        

        
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
            
        print 'True classes:', y.shape, 'Predicted classes:', y_predict.shape
            
        sig_predicted = deepcopy(y_predict)[y==0]
        bkg_predicted = deepcopy(y_predict)[y!=0]
        bkg1_predicted= deepcopy(y_predict)[y==1]
        bkg2_predicted= deepcopy(y_predict)[y==2]
        bkg3_predicted= deepcopy(y_predict)[y==3]
        bkg1_w = deepcopy(w)[y==1]
        bkg2_w = deepcopy(w)[y==2]
        bkg3_w = deepcopy(w)[y==3]
        
        variables = nvar
        
        print 'met threshold:', met_threshold
        plot_output_score(sig_predicted[:,0][X[:,variables.index('met')][y==0]>=met_threshold], sig_w[X[:,variables.index('met')][y==0]>=met_threshold], bkg_predicted[:,0][X[:,variables.index('met')][y!=0]>=met_threshold], bkg_w[X[:,variables.index('met')][y!=0]>=met_threshold], binning, save=save, fileName=fileName, addStr=addStr+'_'+met_cut_addStr, sample=sample,log=True)
        
        plot_output_score_multiclass.plot_output_score_multiclass(sig_predicted[:,0], sig_w, bkg1_predicted[:,0], bkg1_w, bkg2_predicted[:,0], bkg2_w, bkg3_predicted[:,0], bkg3_w, bkg_predicted[:,0], bkg_w, binning, fileName=fileName, save=save, log=True, sample=sample, addStr=addStr+'_'+met_cut_addStr)

        print 'met threshold:', str(250e3)
        plot_output_score(sig_predicted[:,0][X[:,variables.index('met')][y==0]>=250e3], sig_w[X[:,variables.index('met')][y==0]>=250e3], bkg_predicted[:,0][X[:,variables.index('met')][y!=0]>=250e3], bkg_w[X[:,variables.index('met')][y!=0]>=250e3], binning, save=save, fileName=fileName, addStr=addStr+'_met250',log=True,sample=sample)
        
        plot_output_score_multiclass.plot_output_score_multiclass(sig_predicted[:,0][X[:,variables.index('met')][y==0]>=250e3], sig_w[X[:,variables.index('met')][y==0]>=250e3], bkg1_predicted[:,0][X[:,variables.index('met')][y==1]>=250e3], bkg1_w[X[:,variables.index('met')][y==1]>=250e3], bkg2_predicted[:,0][X[:,variables.index('met')][y==2]>=250e3], bkg2_w[X[:,variables.index('met')][y==2]>=250e3], bkg3_predicted[:,0][X[:,variables.index('met')][y==3]>=250e3], bkg3_w[X[:,variables.index('met')][y==3]>=250e3], bkg_predicted[:,0][X[:,variables.index('met')][y!=0]>=250e3], bkg_w[X[:,variables.index('met')][y!=0]>=250e3], binning, fileName=fileName, save=save, log=True, sample=sample, addStr=addStr+'_met250')
        
        
