setupATLAS -q
lsetup "root 6.10.04-x86_64-slc6-gcc62-opt"

export WorkDir=`pwd`
export PYTHONPATH=$PYTHONPATH:$WorkDir/python
export PATH="/home/d/David.Handl/miniconda2/bin:$PATH"
export PATH=$PATH:$WorkDir/scripts
source activate testenv

