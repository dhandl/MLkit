#!/bin/bash
filename=$1

filelines=`cat $filename`
for line in $filelines ; do
  prepareSample.py /project/etp3/dhandl/samples/SUSY/Stop1L/MC15_signals/$line /project/etp5/dhandl/samples/SUSY/Stop1L/FullRun2/hdf5/MC15_signals/$line    
done

