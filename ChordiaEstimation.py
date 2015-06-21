# -*- coding: utf-8 -*-
import numpy as np

import ModeFunctions as mf
import PitchDistribution as p_d

class ChordiaEstimation:

	def __init__(self, cent_ss=7.5, smooth_factor=7.5, chunk_size=60, threshold=0.5):
		self.cent_ss = cent_ss
		self.smooth_factor = smooth_factor
		self.chunk_size = chunk_size
		self.hop_size = 128
		self.threshold = threshold

	def train(self, mode_name, pt_list, ref_freq_list, metric='pcd', save_dir='./'):
		for pt in pt_list:
			cur = mf.load_track(pt, pt_dir)
			time_track = cur[:,0]
			pitch_track = cur[:,1]
			save_name = mode_name + '_' + metric + '.json'
			pts, segs = self.slice(time_track, pitch_track, pt)
			self.train_segments(pts, segs, ref_freq_list, save_dir, save_name)

	def slice(self, time_track, pitch_track, pt_source):
		segments = []
		seg_lims = []
		last = 0
		for k in np.arange(1, (int(max(time_track) / self.chunk_size) + 1)):
			cur = 1 + max(np.where(time_track < self.chunk_size * k)[0])
			segments.append(pitch_track[last:(cur-1)])
			seg_lims.append((pt_source, int(time_track[last]),int(time_track[cur-1])))
			last = cur
		if(len(pitch_track[last:]) >= (self.chunk_size * self.threshold)):
			segments.append(pitch_track[last:])
		return segments, seg_lims

	def train_segments(self, pts, seg_tuples, ref_freq_list, save_dir, save_name):
		cnt = 0
		dist_list = 0
		for idx in range(len(pt_segs)):
			src = seg_tuples[idx][0]
			interval = '(' + str(seg_tuples[1]) + ', ' + str(seg_tuples[2]) + ')'
			dist = mf.generate_pd(pt_segs[idx], ref_freq=ref_freq_list[idx], smooth_factor=self.smooth_factor, cent_ss=self.cent_ss, source=src, segment=interval)
			if(metric=='pcd'):
				dist = generate_pcd(dist)
			dist_list.append(dist)
			
		with open(save_name, 'a') as f:
			f.close()
			for d in dist_list:
				d.save(save_name, save_dir)

		