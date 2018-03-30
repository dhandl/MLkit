import os

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize

import AtlasStyle_mpl

def plot_event_display(content, output_fname=None, vmin=1e-1, vmax=1000e3, title=''):
  '''
  Function to help you visualize an event grid topology on a log scale
  Args:
  -----
      content : numpy array, first arg to imshow, 
          content of the image
          e.g.: images.mean(axis=0) --> the average image

      output_fname : string, name of the output file where the plot will be 
          saved. 

      vmin : (default = 1e-1) float, lower bound of the pixel intensity 
          scale before saturation

      vmax : (default = 1000e3) float, upper bound of the pixel intensity 
          scale before saturation

      title : (default = '') string, title of the plot, to be displayed 
          on top of the image
  '''
  fig, ax = plt.subplots(figsize=(8, 6))
  extent = (-3.2, 3.2, -3, 3)

  im = ax.imshow(content, interpolation='nearest',
                 norm=LogNorm(vmin=vmin, vmax=vmax), origin='lower', extent=extent)
                 #norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent)

  cbar = plt.colorbar(im, fraction=0.05, pad=0.05)
  cbar.set_label(r'Pixel $p_T$ [GeV]', y=0.85)
  plt.xlabel(r'Azimuthal Angle $(\phi)$')
  plt.ylabel(r'Pseudorapidity $(\eta)$')
  plt.title(title)
  AtlasStyle_mpl.ATLASLabel(ax, 0.02, 0.9, 'Work in progress')

  if not output_fname is None:
    plt.savefig(output_fname)
