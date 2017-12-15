#!/bin/bash

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh

export WorkDir=`pwd`
export PYTHONPATH=$PYTHONPATH:$WorkDir/python
export PATH=$PATH:$WorkDir/scripts

if [[ "$HOSTNAME" == *"lxplus"* ]]; then
  echo "Setting up environment on lxplus..."
  export PATH=/afs/cern.ch/work/d/dhandl/public/miniconda3/bin:$PATH
elif [[ "$HOSTNAME" == *"etp"* ]]; then
  echo "Setting up environment on etp..."
  export PATH=/project/etp5/dhandl/miniconda3/bin:$PATH
else
  echo "Setting up environment on c2pap..."
 export PATH=/gpfs/scratch/pr62re/di36jop/workareas/miniconda3/bin:$PATH
fi

source activate testenv
