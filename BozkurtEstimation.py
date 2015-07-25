# -*- coding: utf-8 -*-
import numpy as np
import math

import ModeFunctions as mf
import PitchDistribution as p_d

class BozkurtEstimation:

	def __init__(self, cent_ss=7.5, smooth_factor=7.5, chunk_size=0):
		self.smooth_factor = smooth_factor
		self.cent_ss = cent_ss
		self.chunk_size = chunk_size

	def train(self, mode_name, pt_list, ref_freq_list, pt_dir='./', metric='pcd', save_dir='./'):
		"""---------------------------------------------------------------------------------------
		This function handles everything related to supervised learning portion of this system. 
		It expects the list of text files containing the pitch tracks of the dataset, the array
		of their known tonics and generates the joint distribution of the mode and saves it.
		---------------------------------------------------------------------------------------"""
		mode_track = []
		for idx in range(len(pt_list)):
			if (self.chunk_size == 0):
				cur_track = mf.load_track(pt_list[idx], pt_dir)[:,1]
				cur_cent_track = mf.hz_to_cent(cur_track, ref_freq=ref_freq_list[idx])
				joint_seg = 'all'
			else:
				tmp_track = mf.load_track(pt_list[idx], pt_dir)[:,1]
				time_track = mf.load_track(pt_list[idx], pt_dir)[:,0]
				cur_track, segs = mf.slice(time_track, tmp_track, mode_name, self.chunk_size)
				cur_cent_track = mf.hz_to_cent(cur_track[0], ref_freq=ref_freq_list[idx])
				joint_seg = (segs[0][1], segs[0][2])
			for i in cur_cent_track:
				mode_track.append(i)
		joint_dist = mf.generate_pd(mode_track, smooth_factor=self.smooth_factor, cent_ss=self.cent_ss, source=pt_list, segment=joint_seg)
		if(metric=='pcd'):
			joint_dist = mf.generate_pcd(joint_dist)
		joint_dist.save((mode_name + '.json'), save_dir=save_dir)

	def estimate(self, pitch_track, time_track, mode_names=[], mode_name='', mode_dir='./', est_tonic=True, est_mode=True, rank=1, distance_method="euclidean", metric='pcd', ref_freq=440):
		"""---------------------------------------------------------------------------------------
		This is the high level function that users are expected to interact with, for estimation
		purposes. Using the est_* flags, it is possible to estimate tonic, mode or both.
		---------------------------------------------------------------------------------------"""
		### Preliminaries before the estimations
		if (self.chunk_size > 0):
			cur_track, segs = mf.slice(time_track, tmp_track, mode_name, self.chunk_size)
			cent_track = mf.hz_to_cent(cur_track[0], ref_freq=ref_freq)
		else:
			cent_track = mf.hz_to_cent(pitch_track, ref_freq=ref_freq)
		dist = mf.generate_pd(cent_track, ref_freq=ref_freq, smooth_factor=self.smooth_factor, cent_ss=self.cent_ss)
		dist = mf.generate_pcd(dist) if (metric=='pcd') else dist
		mode_dists = [(p_d.load((m + '.json'), mode_dir)) for m in mode_names]
		mode_dist = p_d.load((mode_name + '_' + metric + '.json'), mode_dir) if (mode_name!='') else None
		tonic_list = np.zeros(min(rank, len(peak_idxs)))
		mode_list = ['' for x in range(min(rank, len(peak_idxs)))]

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
				for m in range(len(mode_dists)):
					dist_mat[:,m] = mf.tonic_estimate(dist, shift_idxs, mode_dists[m], distance_method=distance_method, metric=metric, cent_ss=self.cent_ss)
			
			for r in range(min(rank, len(peak_idxs))):
				min_row = np.where((dist_mat == np.amin(dist_mat)))[0][0]
				min_col = np.where((dist_mat == np.amin(dist_mat)))[1][0]
				if(metric=='pcd'):
					tonic_list[r] = mf.cent_to_hz([dist.bins[peak_idxs[min_row]]], anti_freq)[0]
				elif(metric=='pd'):
					tonic_list[r] = mf.cent_to_hz([shift_idxs[min_row] * self.cent_ss], ref_freq)[0]
				mode_list[r] = mode_names[min_col]
				dist_mat[min_row][min_col] = (np.amax(dist_mat) + 1)
			return mode_list, tonic_list

		elif(est_tonic):
			peak_idxs = shift_idxs if (metric=='pd') else peak_idxs
			ref_freq = anti_freq if (metric=='pcd') else ref_freq
			distance_vector = mf.tonic_estimate(dist, peak_idxs, mode_dist, distance_method=distance_method, metric=metric, cent_ss=self.cent_ss)
			
			for r in range(min(rank, len(peak_idxs))):
				idx = np.argmin(distance_vector)
				if(metric=='pcd'):
					tonic_list[r] = mf.cent_to_hz([dist.bins[peak_idxs[idx]]], anti_freq)[0]
				elif(metric=='pd'):
					tonic_list[r] = mf.cent_to_hz([shift_idxs[idx] * self.cent_ss], ref_freq)[0]
				distance_vector[idx] = (np.amax(distance_vector) + 1)
			return tonic_list

		elif(est_mode):
			distance_vector = mf.mode_estimate(dist, mode_dists, distance_method=distance_method, metric=metric, cent_ss=self.cent_ss)
			for r in range(min(rank, len(peak_idxs))):
				idx = np.argmin(distance_vector)
				mode_list[r] = mode_names[idx]
				distance_vector[idx] = (np.amax(distance_vector) + 1)
			return mode_list
	
		else:
			# Nothing is expected to be estimated
			return 0