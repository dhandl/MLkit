from collections import namedtuple

input = '/gpfs/scratch/pr62re/di36jop/samples/ttbar_rew/TRUTH3/hdf5/'

# define your samples here
Signal = [
  {'name':'powheg_ttbar_TRUTH3', 'path':input+'powheg_ttbar_TRUTH3/'}
]

Background = [
#  {'name':'ttbar_radHi_TRUTH3',  'path':input},
  {'name':'ttbar_radLo_TRUTH3', 'path':input+'ttbar_radLo_TRUTH3/'}
]
