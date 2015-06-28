import numpy as np
import PitchDistribution as p_d
import ModeFunctions as mf
import ChordiaEstimation as ce
import matplotlib.pyplot as pl

pt_dir = 'Examples/Pitch Tracks/'
pd_dir = 'Examples/PD/'
pcd_dir = 'Examples/PCD/'
mode_dir = 'Examples/Chordia Examples/'

c = ce.ChordiaEstimation()

pt1 = np.loadtxt((pt_dir + 'semahat.txt'))[:,1]
segments, seg_lims = c.slice(np.loadtxt((pt_dir + 'semahat.txt'))[:,0], pt1, 'a')
ct = mf.hz_to_cent(segments[0], 199)
dist = mf.generate_pd(ct, ref_freq=199, smooth_factor=7.5, cent_ss=7.5)
dist = mf.generate_pcd(dist)


#c.train('gec_kalma', ['gec_kalma'], [334.9488], metric='pd', save_dir=mode_dir, pt_dir=pt_dir)
#c.train('semahat', ['semahat'], [199], metric='pd', save_dir=mode_dir, pt_dir=pt_dir)
#c.train('murat_derya', ['murat_derya'], [396.3525], metric='pd', save_dir=mode_dir, pt_dir=pt_dir)
print c.estimate(pt1, mode_names=['gec_kalma', 'murat_derya', 'semahat'], mode_name='', mode_dir=mode_dir, est_tonic=False, est_mode=True, rank=6, distance_method='bhat', metric='pcd', ref_freq=199)