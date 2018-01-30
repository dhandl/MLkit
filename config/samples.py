from collections import namedtuple

# the .h5 files used for training and testing are prepared here
 
input = "/gpfs/scratch/pr62re/di36jop/samples/ttbar_rew/TRUTH3/"

# define your samples here
Signal = [
  {'name':'powheg_ttbar_TRUTH3', 'tree':'powheg_ttbar_TRUTH3_Nom', 'path':input}
]

Background = [
  {'name':'ttbar_radHi_TRUTH3', 'tree':'ttbar_radHi_TRUTH3_Nom', 'path':input},
  {'name':'ttbar_radLo_TRUTH3', 'tree':'ttbar_radLo_TRUTH3_Nom', 'path':input}
]
