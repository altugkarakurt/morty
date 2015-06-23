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

	def train(self, mode_name, pt_list, ref_freq_list, metric='pcd', save_dir='./', pt_dir='./'):
		save_name = mode_name + '_' + metric + '.json'
		dist_list = []
		dist_json = []
		for pt in range(len(pt_list)):
			cur = mf.load_track(pt_list[pt], pt_dir)
			time_track = cur[:,0]
			pitch_track = cur[:,1]
			pts, segs = self.slice(time_track, pitch_track, pt_list[pt])
			pts = [mf.hz_to_cent(k, ref_freq=ref_freq_list[pt]) for k in pts]
			temp_list = self.train_segments(pts, segs, ref_freq_list[pt], save_dir, save_name, metric)
			for tmp in temp_list:
				dist_list.append(tmp)

		for d in dist_list:
			temp_json = {'bins':d.bins.tolist(), 'vals':d.vals.tolist(), 'kernel_width':d.kernel_width, 'source':d.source, 'ref_freq':d.ref_freq, 'segmentation':d.segmentation}
			dist_json.append(temp_json)

		with open((save_dir + save_name), 'w') as f:
			dist_json = {mode_name:dist_json}
			json.dump(dist_json, f, indent=2)
			f.close()

	def estimate(self, pitch_track, mode_names=[], mode_name='', mode_dir='./', est_tonic=True, est_mode=True, rank=1, distance_method="euclidean", metric='pcd', ref_freq=440):
		### Preliminaries before the estimations
		cent_track = mf.hz_to_cent(pitch_track, ref_freq)
		dist = mf.generate_pd(cent_track, ref_freq=ref_freq, smooth_factor=self.smooth_factor, cent_ss=self.cent_ss)
		dist = mf.generate_pcd(dist) if (metric=='pcd') else dist
		mode_collections = [self.load_collection(mode, metric, dist_dir=mode_dir) for mode in mode_names]
		mode_dists = [dist for col in mode_collections for dist in col]
		mode_dist = self.load_collection(mode_name, metric, dist_dir=mode_dir) if (mode_name!='') else None
		tonic_list = np.zeros(rank)
		mode_list = ['' for x in range(rank)]

		if(est_tonic):
			if(metric=='pcd'):
				### Shifting to the global minima to prevent wrong detection of peaks
				shift_factor = dist.vals.tolist().index(min(dist.vals))
				dist = dist.shift(shift_factor)
				anti_freq = mf.cent_to_hz([dist.bins[shift_factor]], ref_freq=ref_freq)[0]
				peak_idxs, peak_vals = dist.detect_peaks()
			elif(metric=='pd'):
				peak_idxs, peak_vals = dist.detect_peaks()
				origin =  np.where(dist.bins==0)[0][0]
				shift_idxs = [(idx - origin) for idx in peak_idxs]

		### Call to actual estimation functions
		if(est_tonic and est_mode):
			if(metric=='pcd'):
				dist_mat = mf.generate_distance_matrix(dist, peak_idxs, mode_dists, method=distance_method)
				for r in range(rank):
					min_row = np.where((dist_mat == np.amin(dist_mat)))[0][0]
					min_col = np.where((dist_mat == np.amin(dist_mat)))[1][0]
					tonic_list[r] = mf.cent_to_hz([dist.bins[peak_idxs[min_row]]], anti_freq)[0]
					mode_list[r] = (mode_dists[min_col].source, mode_dists[min_col].segmentation)
					dist_mat[min_row][min_col] = (np.amax(dist_mat) + 1)
				return mode_list, tonic_list

			elif(metric=='pd'):
				dist_mat = np.zeros((len(shift_idxs), len(mode_dists)))
				for m in range(len(mode_dists)):
					dist_mat[:,m] = self.tonic_estimate(dist, shift_idxs, mode_dists[m], distance_method=distance_method, metric=metric)
				for r in range(rank):
					min_row = np.where((dist_mat == np.amin(dist_mat)))[0][0]
					min_col = np.where((dist_mat == np.amin(dist_mat)))[1][0]
					tonic_list[r] = mf.cent_to_hz([shift_idxs[min_row] * self.cent_ss], ref_freq)[0]
					mode_list[r] = (mode_dists[min_col].source, mode_dists[min_col].segmentation)
					dist_mat[min_row][min_col] = (np.amax(dist_mat) + 1)
				return mode_list, tonic_list

		elif(est_tonic):
			if(metric=='pcd'):
				dist_mat = [(np.array(mf.generate_distance_matrix(dist, peak_idxs, [d], method=distance_method))[:,0]) for d in mode_dist]

			elif(metric=='pd'):
				peak_idxs = shift_idxs
				temp = p_d.PitchDistribution(dist.bins, dist.vals, kernel_width=dist.kernel_width, source=dist.source, ref_freq=dist.ref_freq, segment=dist.segmentation)
				dist_mat = []
				for d in mode_dist:
					temp, d = mf.pd_zero_pad(temp, d, cent_ss=self.cent_ss)

					### Filling both sides of vals with zeros, to make sure that the shifts won't drop any non-zero values
					temp.vals = np.concatenate((np.zeros(abs(max(peak_idxs))), temp.vals, np.zeros(abs(min(peak_idxs)))))
					d.vals = np.concatenate((np.zeros(abs(max(peak_idxs))), d.vals, np.zeros(abs(min(peak_idxs)))))
					cur_vector = np.array(mf.generate_distance_matrix(temp, peak_idxs, [d], method=distance_method))[:,0]
					dist_mat.append(cur_vector)

			for r in range(rank):
				min_row = np.where((dist_mat == np.amin(dist_mat)))[0][0]
				min_col = np.where((dist_mat == np.amin(dist_mat)))[1][0]
				tonic_list[r] = mf.cent_to_hz([dist.bins[peak_idxs[min_row]]], anti_freq)[0]
				dist_mat[min_row][min_col] = (np.amax(dist_mat) + 1)
			return tonic_list

		elif(est_mode):
			distance_vector = self.mode_estimate(dist, mode_dists, distance_method=distance_method, metric=metric)
			for r in range(rank):
				idx = np.argmin(distance_vector)
				mode_list[r] = (mode_dists[idx].source, mode_dists[idx].segmentation)
				distance_vector[idx] = (np.amax(distance_vector) + 1)
			return mode_list

		else:
			return 0
		
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
			seg_lims.append((pt_source, int(round(time_track[last])), int(round(time_track[len(time_track) - 1]))))
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

	def load_collection(self, mode_name, metric, dist_dir='./'):
		obj_list = []
		fname = mode_name + '_' + metric + '.json'
		with open((dist_dir + fname)) as f:
			dist_list = json.load(f)[mode_name]
		for d in dist_list:
			obj_list.append(p_d.PitchDistribution(np.array(d['bins']), np.array(d['vals']), kernel_width=d['kernel_width'], source=d['source'], ref_freq=d['ref_freq'], segment=d['segmentation']))
		return obj_list

		