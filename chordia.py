import numpy as np
import PitchDistribution as p_d
import ModeFunctions as mf
import ChordiaEstimation as ce

pt_dir = 'Examples/Pitch Tracks/'
pd_dir = 'Examples/PD/'
pcd_dir = 'Examples/PCD/'
mode_dir = 'Examples/Chordia Examples/'

c = ce.ChordiaEstimation()

pt1 = mf.load_track('murat_derya', pt_dir)[:,1]

print c.estimate(pt1, mode_names=[], mode_name='ussak', mode_dir=mode_dir, est_tonic=True, est_mode=False, rank=1, distance_method="euclidean", metric='pcd', ref_freq=440)
