# README

This is the framework for ongoing ML R&D studies for the ttbar+MET 1L analysis.

### Getting started

Check your global config before with 
```
git config --list
```

Set up your config properly:
```
git config --global user.name "Firstname Lastname"
git config --global user.email example@example.com
```

```
git clone ssh://git@gitlab.cern.ch:7999/dhandl/stop1l-MLkit.git
cd stop1l-MLkit
source setup.sh #(this command will eventually not work for you!)
```

More info will follow. 

### Miscellanea

Necessary python modules are loaded via an Anaconda based environment. Find a list of installed packages here:
root, python, mkl, jupyter, numpy, scipy, matplotlib, scikit-learn, h5py, rootpy, root-numpy, pandas, scikit-image, seaborn, mkl-service, tqdm

Might be that root is not working properly. In this case try:

```
source <PATH_TO_YOUR_CONDA_ENVIRONMENT>/bin/thisroot.sh #to check your available environments try conda info --envs
``` 
