#!/bin/bash

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh

if [[ "$HOSTNAME" == *"lxplus"* ]]; then
  echo "Setting up environment on lxplus..."
  export PATH=/afs/cern.ch/work/d/dhandl/public/miniconda3/bin:$PATH
elif [[ "$HOSTNAME" == *"etp"* ]]; then
  echo "Setting up environment on etp..."
  export PATH=/project/etp5/dhandl/miniconda3/bin:$PATH
  CondaDir="/project/etp5/dhandl/miniconda3"
else
  echo "Setting up environment on c2pap..."
  export PATH=/gpfs/scratch/pr62re/di36jop/workareas/miniconda3/bin:$PATH
  CondaDir="/gpfs/scratch/pr62re/di36jop/workareas/miniconda3"
fi

export WorkDir=`pwd`
export PYTHONPATH=$PYTHONPATH:$WorkDir/python
export PYTHONPATH=$PYTHONPATH:$WorkDir/config
export PATH=$PATH:$WorkDir/scripts

source activate testenv
source "$CondaDir/envs/testenv/bin/thisroot.sh"
#export MKL_THREADING_LAYER=GNU
