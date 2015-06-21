# -*- coding: utf-8 -*-
import numpy as np
import json
import ModeFunctions as mf
import PitchDistribution as p_d

txtlist = ['semahat', 'gec_kalma', 'murat_derya']
freqlist = [199, 396.3525, 334.9488]

for t in range(len(txtlist)):
	hz_track = np.loadtxt(txtlist[t] + '.txt')[:,1]
	cent_track = mf.hz_to_cent(hz_track, ref_freq=freqlist[t])
	pd = mf.generate_pd(cent_track, ref_freq=freqlist[t], smooth_factor=7.5, cent_ss=7.5, source=txtlist[t])
	pcd = mf.generate_pcd(pd)
	pd.save((txtlist[t] + '_pd.json'))
	pcd.save((txtlist[t] + '_pcd.json'))