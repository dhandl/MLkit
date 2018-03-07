import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

import matplotlib.pyplot as plt

def ATLASLabel(x, y, text='Work in progress', color='black'):
  atlas = plt.text(x, y, 'ATLAS', style='italic', weight='bold', size='x-large', color=color)
  add = plt.text(x, y, text, size='x-large', color=color)

def LumiLabel(x, y, lumi=100, color='black'):
  plt.text(x, y, '$\sqrt{\mathrm{s}}=$13 TeV, L$=$'+str(lumi)+'fb$^{-1}$', size='x-large')
