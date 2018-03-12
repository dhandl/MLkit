import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

import matplotlib.pyplot as plt

def ATLASLabel(axes, x, y, text='Work in progress', color='black'):
  #bb = dict(boxstyle='square', fc='w', lw=0)
  atlas = axes.text(x, y, 'ATLAS', style='italic', weight='bold', size='x-large', color=color, transform=axes.transAxes)
  p = atlas.get_position()
  add = axes.text(x+0.12, y, text, size='x-large', color=color, transform=axes.transAxes)

def LumiLabel(axes, x, y, lumi=100, color='black'):
  axes.text(x, y, '$\sqrt{\mathrm{s}}=$13 TeV, L$=$'+str(lumi)+'fb$^{-1}$', size='x-large', transform=axes.transAxes)
