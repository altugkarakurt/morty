# -*- coding: utf-8 -*-
import numpy as np
import ModeFunctions as mf
import PitchDistribution as p_d
import json
import pdb
import os

class ChordiaEstimation:

	def __init__(self, cent_ss=7.5, smooth_factor=7.5, chunk_size=60, threshold=0.5, overlap=0):
		self.cent_ss = cent_ss
		self.overlap = overlap
		self.smooth_factor = smooth_factor
		self.chunk_size = chunk_size
		self.threshold = threshold

	def train(self, mode_name, pt_list, ref_freq_list, metric='pcd', save_dir='./', pt_dir='./'):
		save_name = mode_name + '.json'
		dist_list = []
		for pt in range(len(pt_list)):
			cur = mf.load_track(pt_list[pt], pt_dir)
			time_track = cur[:,0]
			pitch_track = cur[:,1]
			pts, segs = mf.slice(time_track, pitch_track, pt_list[pt], self.chunk_size, self.threshold, self.overlap)
			pts = [mf.hz_to_cent(k, ref_freq=ref_freq_list[pt]) for k in pts]
			temp_list = self.train_segments(pts, segs, ref_freq_list[pt], save_dir, save_name, metric)
			for tmp in temp_list:
				dist_list.append(tmp)

		dist_json = [{'bins':d.bins.tolist(), 'vals':d.vals.tolist(), 'kernel_width':d.kernel_width, 'source':d.source, 'ref_freq':d.ref_freq, 'segmentation':d.segmentation, 'overlap':d.overlap} for d in dist_list]


		with open(os.path.join(save_dir, save_name), 'w') as f:
			json.dump(dist_json, f, indent=2)
			f.close()

	def estimate(self, pitch_track, time_track, mode_names=[], mode_name='', mode_dir='./', est_tonic=True, est_mode=True, distance_method="euclidean", metric='pcd', ref_freq=440, k_param=1):
		pts, segs = mf.slice(time_track, pitch_track, 'input', self.chunk_size, self.threshold, self.overlap)
		tonic_list = 0
		mode_list = ''
		min_cnt = len(pts) * k_param
		if(est_tonic and est_mode):
			neighbors = [ [mode_list, tonic_list] for i in range(len(segs)) ]
		elif(est_tonic):
			neighbors = [ tonic_list for i in range(len(segs)) ]
		elif(est_mode):
			neighbors = [ mode_list for i in range(len(segs)) ]

		for p in range(len(pts)):
			neighbors[p] = self.segment_estimate(pts[p], mode_names=mode_names, mode_name=mode_name, mode_dir=mode_dir, est_tonic=est_tonic, est_mode=est_mode, distance_method=distance_method, metric=metric, ref_freq=ref_freq, min_cnt=min_cnt)
		
		candidate_distances, candidate_ests, candidate_sources, kn_ests, kn_sources, idx_counts, elem_counts = ([] for i in range(7))

		if(est_mode and est_tonic):
			for i in xrange(len(pts)):
				for j in neighbors[i][1]:
					candidate_distances.append(j)
				for l in xrange(len(neighbors[i][0][1])):
					candidate_ests.append((neighbors[i][0][1][l], neighbors[i][0][0][l][0]))
					candidate_sources.append(neighbors[i][0][0][l][1])

			for k in xrange(k_param):
				idx = np.argmin(candidate_distances)
				kn_ests.append(candidate_ests[idx])
				kn_sources.append(candidate_sources[idx])
				candidate_distances[idx] = (np.amax(candidate_distances) + 1)
			
			for c in set(kn_ests):
				idx_counts.append(kn_ests.count(c))
				elem_counts.append(c)
			joint_estimation = elem_counts[np.argmax(idx_counts)]

			for m in xrange(len(kn_ests)):
				if (not (kn_ests[m] == joint_estimation)):
					kn_sources.remove(kn_sources[m])
			result = [joint_estimation, kn_sources]

		elif(est_mode):
			for i in xrange(len(pts)):
				for j in neighbors[i][1]:
					candidate_distances.append(j)
				for l in xrange(len(neighbors[i][0])):
					candidate_ests.append(neighbors[i][0][l][0])
					candidate_sources.append(neighbors[i][0][l][1])

			for k in xrange(k_param):
				idx = np.argmin(candidate_distances)
				kn_ests.append(candidate_ests[idx])
				kn_sources.append(candidate_sources[idx])
				candidate_distances[idx] = (np.amax(candidate_distances) + 1)
			
			for c in set(kn_ests):
				idx_counts.append(kn_ests.count(c))
				elem_counts.append(c)
			mode_estimation = elem_counts[np.argmax(idx_counts)]

			for m in xrange(len(kn_ests)):
				if (not (kn_ests[m] == mode_estimation)):
					kn_sources.remove(kn_sources[m])
			result = [mode_estimation, kn_sources]

		elif(est_tonic):

			for i in xrange(len(pts)):
				for j in neighbors[i][1]:
					candidate_distances.append(j)
				for l in xrange(len(neighbors[i][0])):
					candidate_ests.append(neighbors[i][0][l][0])
					candidate_sources.append(neighbors[i][0][l][1])

			for k in xrange(k_param):
				idx = np.argmin(candidate_distances)
				kn_ests.append(candidate_ests[idx])
				kn_sources.append(candidate_sources[idx])
				candidate_distances[idx] = (np.amax(candidate_distances) + 1)

			for c in set(kn_ests):
				idx_counts.append(kn_ests.count(c))
				elem_counts.append(c)
			tonic_estimation = elem_counts[np.argmax(idx_counts)]

			for m in xrange(len(kn_ests)):
				if (not (kn_ests[m] == tonic_estimation)):
					kn_sources.remove(kn_sources[m])
			result = [tonic_estimation, kn_sources]

		print result
		return 0

	def segment_estimate(self, pitch_track, mode_names=[], mode_name='', mode_dir='./', est_tonic=True, est_mode=True, distance_method="euclidean", metric='pcd', ref_freq=440, min_cnt=3):
		### Preliminaries before the estimations
		cent_track = mf.hz_to_cent(pitch_track, ref_freq)
		dist = mf.generate_pd(cent_track, ref_freq=ref_freq, smooth_factor=self.smooth_factor, cent_ss=self.cent_ss)
		dist = mf.generate_pcd(dist) if (metric=='pcd') else dist
		mode_collections = [self.load_collection(mode, metric, dist_dir=mode_dir) for mode in mode_names]
		cum_lens = np.cumsum([len(col) for col in mode_collections])
		mode_dists = [d for col in mode_collections for d in col]
		mode_dist = self.load_collection(mode_name, metric, dist_dir=mode_dir) if (mode_name!='') else None
		tonic_list = [0 for x in range(min_cnt)]
		mode_list = ['' for x in range(min_cnt)]
		min_distance_list = np.zeros(min_cnt)

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
			elif(metric=='pd'):
				dist_mat = np.zeros((len(shift_idxs), len(mode_dists)))
				for m in xrange(len(mode_dists)):
					dist_mat[:,m] = mf.tonic_estimate(dist, shift_idxs, mode_dists[m], distance_method=distance_method, metric=metric, cent_ss=self.cent_ss)

			for r in xrange(min_cnt):
				min_row = np.where((dist_mat == np.amin(dist_mat)))[0][0]
				min_col = np.where((dist_mat == np.amin(dist_mat)))[1][0]	
				if(metric=='pcd'):
					tonic_list[r] = mf.cent_to_hz([dist.bins[peak_idxs[min_row]]], anti_freq)[0]
				elif(metric=='pd'):
					tonic_list[r] = mf.cent_to_hz([shift_idxs[min_row] * self.cent_ss], ref_freq)[0]
				mode_list[r] = (mode_names[min(np.where((cum_lens > min_col))[0])], mode_dists[min_col].source[:-6])
				min_distance_list[r] = dist_mat[min_row][min_col]
				dist_mat[min_row][min_col] = (np.amax(dist_mat) + 1)
			return [[mode_list, tonic_list], min_distance_list.tolist()]

		elif(est_tonic):
			peak_idxs = shift_idxs if metric=='pd' else peak_idxs
			dist_mat = [mf.tonic_estimate(dist, peak_idxs, d, distance_method=distance_method, metric=metric, cent_ss=self.cent_ss) for d in mode_dist]
			anti_freq = ref_freq if metric=='pd' else anti_freq

			for r in xrange(min_cnt):
				min_row = np.where((dist_mat == np.amin(dist_mat)))[0][0]
				min_col = np.where((dist_mat == np.amin(dist_mat)))[1][0]
				tonic_list[r] = (mf.cent_to_hz([dist.bins[peak_idxs[min_col]]], anti_freq)[0], mode_dists[min_row].source[:-6])
				min_distance_list[r] = dist_mat[min_row][min_col]
				dist_mat[min_row][min_col] = (np.amax(dist_mat) + 1)
			return [tonic_list, min_distance_list.tolist()]

		elif(est_mode):
			distance_vector = mf.mode_estimate(dist, mode_dists, distance_method=distance_method, metric=metric, cent_ss=self.cent_ss)
			for r in xrange(min_cnt):
				idx = np.argmin(distance_vector)
				mode_list[r] = (mode_names[min(np.where((cum_lens > idx))[0])], mode_dists[idx].source[:-6])
				min_distance_list[r] = distance_vector[idx]
				distance_vector[idx] = (np.amax(distance_vector) + 1) 
			return [mode_list, min_distance_list.tolist()]

		else:
			return 0

	def train_segments(self, pts, seg_tuples, ref_freq, save_dir, save_name, metric='pcd'):
		dist_list = []
		for idx in range(len(pts)):
			src = seg_tuples[idx][0]
			interval = (seg_tuples[idx][1], seg_tuples[idx][2])
			dist = mf.generate_pd(pts[idx], ref_freq=ref_freq, smooth_factor=self.smooth_factor, cent_ss=self.cent_ss, source=src, segment=interval, overlap=self.overlap)
			if(metric=='pcd'):
				dist = mf.generate_pcd(dist)
			dist_list.append(dist)
		return dist_list

	def load_collection(self, mode_name, metric, dist_dir='./'):
		obj_list = []
		fname = mode_name + '.json'
		with open(os.path.join(dist_dir, fname)) as f:
			dist_list = json.load(f)
		for d in dist_list:
			obj_list.append(p_d.PitchDistribution(np.array(d['bins']), np.array(d['vals']), kernel_width=d['kernel_width'], source=d['source'], ref_freq=d['ref_freq'], segment=d['segmentation'], overlap=d['overlap']))
		return obj_list