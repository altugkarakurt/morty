import numpy as np
import PitchDistribution as p_d
import ModeFunctions as mf
import ChordiaEstimation as ce
import matplotlib.pyplot as pl

pt_dir = 'Examples/Pitch Tracks/'
pd_dir = 'Examples/PD/'
pcd_dir = 'Examples/PCD/'
mode_dir = 'Examples/Chordia Examples/'

pt1 = np.loadtxt((pt_dir + 'semahat.txt'))[:,1]

c = ce.ChordiaEstimation()

#c.train('ussak', ['semahat', 'murat_derya', 'gec_kalma'], [199, 396.3525, 334.9488], metric='pcd', save_dir=mode_dir, pt_dir=pt_dir)

print c.estimate(pt1, mode_names=[], mode_name='ussak', mode_dir=mode_dir, est_tonic=True, est_mode=False, rank=3, distance_method="euclidean", metric='pcd', ref_freq=440)
