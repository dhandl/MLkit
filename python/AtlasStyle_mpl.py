import matplotlib
matplotlib.rcParams['figure.figsize'] = [8, 6]

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

matplotlib.rcParams['axes.titlesize'] = 22
matplotlib.rcParams['axes.labelsize'] = 16

matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['lines.markersize'] = 6

matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['xtick.major.size'] = 8.0
matplotlib.rcParams['xtick.minor.size'] = 4.0
matplotlib.rcParams['xtick.minor.visible'] = True
matplotlib.rcParams['xtick.minor.bottom'] = True
matplotlib.rcParams['xtick.minor.top'] = True

matplotlib.rcParams['ytick.left'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['ytick.major.size'] = 8.0
matplotlib.rcParams['ytick.minor.size'] = 4.0
matplotlib.rcParams['ytick.minor.visible'] = True
matplotlib.rcParams['ytick.minor.left'] = True
matplotlib.rcParams['ytick.minor.right'] = True

matplotlib.rcParams['legend.frameon'] = False
matplotlib.rcParams['legend.fontsize'] = 'large'


matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

import matplotlib.pyplot as plt

def ATLASLabel(axes, x, y, text='Work in progress', color='black'):
  #bb = dict(boxstyle='square', fc='w', lw=0)
  atlas = axes.text(x, y, 'ATLAS', style='italic', weight='bold', size='x-large', color=color, transform=axes.transAxes)
  p = atlas.get_position()
  add = axes.text(x+0.12, y, text, size='x-large', color=color, transform=axes.transAxes)

def LumiLabel(axes, x, y, lumi=100, color='black'):
  axes.text(x, y, '$\sqrt{\mathrm{s}}=$13 TeV, L$=$'+str(lumi)+'fb$^{-1}$', size='x-large', transform=axes.transAxes)
