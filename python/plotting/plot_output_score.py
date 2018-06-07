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

import AtlasStyle_mpl

#Other
import prepareTraining as pT
from collections import namedtuple

Sample = namedtuple('Sample', 'name dataframe')

from sklearn.preprocessing import StandardScaler

def plot_output_score(sig_predicted, sig_w, bkg_predicted, bkg_w, binning, fileName='Test', normed=False, save=False, addStr='', ratio=True, title=None):
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

  s_hist, s_bins, s_patches = plt.hist(sig_predicted.ravel(), weights=sig_w, histtype='stepfilled', color='r', label='signal', alpha=0.5, bins=binning[0], range=(binning[1], binning[2]), density=normed)
  b_hist, b_bins, b_patches = plt.hist(bkg_predicted.ravel(), weights=bkg_w, histtype='stepfilled', color='b', label='background', alpha=0.5, bins=binning[0], range=(binning[1], binning[2]), density=normed)

  s_w = getSumW2(sig_predicted.ravel(), sig_w, binning)
  b_w = getSumW2(bkg_predicted.ravel(), bkg_w, binning)

  #sep = getSeparation(s_histTest, s_binsTest, b_histTest, b_binsTest)

  #print sep

  if normed:
    ax1.set_ylabel('a. u.', ha='left')
  else:
    ax1.set_ylabel('Events', ha='left')
  
  #ax1.set_ylim((0, s_hist.max()*(1+0.33)))
  ax1.set_ylim((0, b_hist.max()))
  leg = plt.legend(loc='best', frameon=False)
  p = leg.get_window_extent()
  #ax.annotate('KS Test S (B): %.3f (%.3f)'%(ks_sig, ks_bkg),(p.p0[0], p.p1[1]), (p.p0[0], p.p1[1]), xycoords='figure pixels', zorder=9)
  #ax1.text(0.65, 0.7, 'KS Test S (B): %.3f (%.3f)'%(ks_sig, ks_bkg), transform=ax1.transAxes)
  #ax1.text(0.65, 0.70, '$<S^2>$ = %.3f'%(sep), transform=ax1.transAxes)
  #ax.text(0.55, 0.7, 'KS p-value S (B): %.3f (%.3f)'%(ks_sig_p, ks_bkg_p), transform=ax.transAxes)
  
  if title is not None:
      plt.title(title)

  AtlasStyle_mpl.ATLASLabel2(ax1, 0.02, 0.9, 'Work in progress')
  AtlasStyle_mpl.LumiLabel(ax1, 0.02, 0.8, lumi=140)
  if ratio:
    ax2 = plt.subplot2grid((4,4), (3,0), colspan=4, rowspan=1)
    r = getRatio(b_hist, b_bins, b_w, s_hist, s_bins, s_w, 'r')
    ax2.set_xlabel('Output score')
    ax2.set_ylabel('B/S')
    ax2.set_xlim((binning[1],binning[2]))
    ax2.set_ylim((-0.5,2.5))
    ax2.grid()
    ax2.tick_params(direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')

  ax1.set(xlabel='Output score')

  if save:
    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
        print('Creating folder plots')
    plt.savefig('plots/'+fileName+'_output_score' + addStr +'.pdf')
    plt.savefig('plots/'+fileName+'_output_score' + addStr +'.png')
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
            
        sig_predicted = y_predict[y==0]
        bkg_predicted= y_predict[y!=0]
        
        variables = nvar
        
        print 'met threshold:', met_threshold
        plot_output_score(sig_predicted[:,0][X[:,variables.index('met')][y==0]>=met_threshold], sig_w[X[:,variables.index('met')][y==0]>=met_threshold], bkg_predicted[:,0][X[:,variables.index('met')][y!=0]>=met_threshold], bkg_w[X[:,variables.index('met')][y!=0]>=met_threshold], binning, save=save, fileName=fileName, addStr=addStr+met_cut_addStr, title=title)

        print 'met threshold:', str(250e3)
        plot_output_score(sig_predicted[:,0][X[:,variables.index('met')][y==0]>=250e3], sig_w[X[:,variables.index('met')][y==0]>=250e3], bkg_predicted[:,0][X[:,variables.index('met')][y!=0]>=250e3], bkg_w[X[:,variables.index('met')][y!=0]>=250e3], binning, save=save, fileName=fileName, addStr=addStr+'met250',title=title)
