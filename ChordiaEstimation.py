# -*- coding: utf-8 -*-
import numpy as np

import ModeFunctions as mf
import PitchDistribution as p_d
import json

class ChordiaEstimation:

	def __init__(self, cent_ss=7.5, smooth_factor=7.5, chunk_size=60, threshold=0.5):
		self.cent_ss = cent_ss
		self.smooth_factor = smooth_factor
		self.chunk_size = chunk_size
		self.threshold = threshold
		self.dummy = 'dummy'

	def train(self, mode_name, pt_list, ref_freq_list, metric='pcd', save_dir='./', pt_dir='./'):
		save_name = mode_name + '_' + metric + '.json'
		dist_list = []
		dist_json = []
		for pt in range(len(pt_list)):
			cur = mf.load_track(pt_list[pt], pt_dir)
			time_track = cur[:,0]
			pitch_track = cur[:,1]
			pts, segs = self.slice(time_track, pitch_track, pt_list[pt])
			temp_list = self.train_segments(pts, segs, ref_freq_list[pt], save_dir, save_name, metric)
			for tmp in temp_list:
				dist_list.append(tmp)

		for d in dist_list:
			temp_json = {'bins':d.bins.tolist(), 'vals':d.vals.tolist(), 'kernel_width':d.kernel_width, 'source':d.source, 'ref_freq':d.ref_freq, 'segmentation':d.segmentation}
			dist_json.append(temp_json)

		with open((save_dir + save_name), 'w') as f:
			dist_json = {self.dummy:dist_json}
			json.dump(dist_json, f, indent=2)
			f.close()

	def slice(self, time_track, pitch_track, pt_source):
		segments = []
		seg_lims = []
		last = 0
		for k in np.arange(1, (int(max(time_track) / self.chunk_size) + 1)):
			cur = 1 + max(np.where(time_track < self.chunk_size * k)[0])
			segments.append(pitch_track[last:(cur-1)])
			seg_lims.append((pt_source, int(round(time_track[last])), int(round(time_track[cur-1])))) #0 - source, 1 - init, 2 - final
			last = cur
		if((max(time_track) - time_track[last]) >= (self.chunk_size * self.threshold)):
			segments.append(pitch_track[last:])
			seg_lims.append((pt_source, round(time_track[last]), round(time_track[len(time_track) - 1])))
		return segments, seg_lims

	def train_segments(self, pts, seg_tuples, ref_freq, save_dir, save_name, metric='pcd'):
		dist_list = []
		for idx in range(len(pts)):
			src = seg_tuples[idx][0]
			interval = (seg_tuples[idx][1], seg_tuples[idx][2])
			dist = mf.generate_pd(pts[idx], ref_freq=ref_freq, smooth_factor=self.smooth_factor, cent_ss=self.cent_ss, source=src, segment=interval)
			if(metric=='pcd'):
				dist = mf.generate_pcd(dist)
			dist_list.append(dist)
		return dist_list

	def load_collection(self, fname, dist_dir='./'):
		obj_list = []
		with open((dist_dir + fname)) as f:
			dist_list = json.load(f)[self.dummy]
		for d in dist_list:
			obj_list.append(p_d.PitchDistribution(np.array(d['bins']), np.array(d['vals']), kernel_width=d['kernel_width'], source=d['source'], ref_freq=d['ref_freq'], segment=d['segmentation']))
		return obj_list

		