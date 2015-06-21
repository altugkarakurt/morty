# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as pl

import ModeFunctions as mf
import PitchDistribution as p_d

class BozkurtEstimation:

	def __init__(self, cent_ss=7.5, smooth_factor=7.5):
		self.smooth_factor = smooth_factor
		self.cent_ss = cent_ss

	def train(self, mode_name, pt_list, ref_freq_list, pt_dir='./', metric='pcd', save_dir='./'):
		"""---------------------------------------------------------------------------------------
		This function handles everything related to supervised learning portion of this system. 
		It expects the list of text files containing the pitch tracks of the dataset, the array
		of their known tonics and generates the joint distribution of the mode and saves it.
		---------------------------------------------------------------------------------------"""
		mode_track = []
		for idx in range(len(pt_list)):
			cur_track = mf.load_track(pt_list[idx], pt_dir)[:,1]
			cur_cent_track = mf.hz_to_cent(cur_track, ref_freq=ref_freq_list[idx])
			for i in cur_cent_track:
				mode_track.append(i)
		joint_dist = mf.generate_pd(mode_track, smooth_factor=self.smooth_factor, cent_ss=self.cent_ss, source=mode_name, segment='all')[0]
		if(metric=='pcd'):
			joint_dist = mf.generate_pcd(joint_dist)
		joint_dist.save((mode_name + '_' + metric + '.json'), save_dir=save_dir)

	def estimate(self, pitch_track, mode_names=[], mode_name='', mode_dir='./', est_tonic=True, est_mode=True, rank=1, distance_method="euclidean", metric='pcd', ref_freq=440):
		"""---------------------------------------------------------------------------------------
		This is the high level function that users are expected to interact with, for estimation
		purposes. Using the est_* flags, it is possible to estimate tonic, mode or both.
		---------------------------------------------------------------------------------------"""
		### Preliminaries before the estimations
		cent_track = mf.hz_to_cent(pitch_track, ref_freq=ref_freq)
		dist = mf.generate_pd(cent_track, ref_freq=ref_freq, smooth_factor=self.smooth_factor, cent_ss=self.cent_ss)[0]
		dist = mf.generate_pcd(dist) if (metric=='pcd') else dist
		mode_dists = [(p_d.load((m + '_' + metric + '.json'), mode_dir)) for m in mode_names]
		mode_dist = p_d.load((mode_name + '_' + metric + '.json'), mode_dir) if (mode_name!='') else None
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
					mode_list[r] = mode_names[min_col]
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
					mode_list[r] = mode_names[min_col]
					dist_mat[min_row][min_col] = (np.amax(dist_mat) + 1)
				return mode_list, tonic_list

		elif(est_tonic):
			if(metric=='pcd'):
				distance_vector = self.tonic_estimate(dist, peak_idxs, mode_dist, distance_method=distance_method, metric=metric)
				for r in range(rank):
					idx = np.argmin(distance_vector)
					tonic_list[r] = mf.cent_to_hz([dist.bins[peak_idxs[idx]]], anti_freq)[0]
					distance_vector[idx] = (np.amax(distance_vector) + 1)
				return tonic_list
			elif(metric=='pd'):
				distance_vector = self.tonic_estimate(dist, shift_idxs, mode_dist, distance_method=distance_method, metric=metric)
				for r in range(rank):
					idx = np.argmin(distance_vector)
					tonic_list[r] = mf.cent_to_hz([shift_idxs[idx] * self.cent_ss], ref_freq)[0]
					distance_vector[idx] = (np.amax(distance_vector) + 1)
				return tonic_list

		elif(est_mode):
			distance_vector = self.mode_estimate(dist, mode_dists, distance_method=distance_method, metric=metric)
			for r in range(rank):
				idx = np.argmin(distance_vector)
				mode_list[r] = mode_names[idx]
				distance_vector[idx] = (np.amax(distance_vector) + 1)
			return mode_list
	
		else:
			# Nothing is expected to be estimated
			return 0

	def tonic_estimate(self, dist, peak_idxs, mode_dist, distance_method="euclidean", metric='pcd'):
		"""---------------------------------------------------------------------------------------
		Given the mode (or candidate mode), compares the piece's distribution using the candidate
		tonics and returns the resultant distance vector to higher level functions.
		---------------------------------------------------------------------------------------"""
		### Mode is known, tonic is estimated.
		### Piece's distributon is generated
		
		if(metric=='pcd'):
			return np.array(mf.generate_distance_matrix(dist, peak_idxs, [mode_dist], method=distance_method))[:,0]

		elif(metric=='pd'):
			temp = p_d.PitchDistribution(dist.bins, dist.vals, kernel_width=dist.kernel_width, source=dist.source, ref_freq=dist.ref_freq, segment=dist.segmentation)
			temp, mode_dist = self.pd_zero_pad(temp, mode_dist)

			### Filling both sides of vals with zeros, to make sure that the shifts won't drop any non-zero values
			temp.vals = np.concatenate((np.zeros(abs(max(peak_idxs))), temp.vals, np.zeros(abs(min(peak_idxs)))))
			mode_dist.vals = np.concatenate((np.zeros(abs(max(peak_idxs))), mode_dist.vals, np.zeros(abs(min(peak_idxs)))))
			
			return np.array(mf.generate_distance_matrix(temp, peak_idxs, [mode_dist], method=distance_method))[:,0]			
	
	def mode_estimate(self, dist, mode_dists, distance_method='euclidean', metric='pcd'):
		"""---------------------------------------------------------------------------------------
		Given the tonic (or candidate tonic), compares the piece's distribution using the candidate
		modes and returns the resultant distance vector to higher level functions.
		---------------------------------------------------------------------------------------"""

		if(metric=='pcd'):
			distance_vector = np.array(mf.generate_distance_matrix(dist, [0], mode_dists, method=distance_method))

		elif(metric=='pd'):
			distance_vector = np.zeros(len(mode_dists))
			for i in range(len(mode_dists)):
				trial = p_d.PitchDistribution(dist.bins, dist.vals, kernel_width=dist.kernel_width, source=dist.source, ref_freq=dist.ref_freq, segment=dist.segmentation)
				trial, mode_trial = self.pd_zero_pad(trial, mode_dists[i])
				distance_vector[i] = mf.distance(trial, mode_trial, method=distance_method)
		return distance_vector

	def pd_zero_pad(self, pd, mode_pd):
		"""---------------------------------------------------------------------------------------
		This function is only used while detecting tonic and working with pd as metric. It pads
		zeros from both sides of the values array to avoid losing non-zero values when comparing
		and to make sure the two PDs are of the same length 
		---------------------------------------------------------------------------------------"""
		### Alignment of the left end-points
		if((min(pd.bins) - min(mode_pd.bins)) > 0):
			temp_left_shift = (min(pd.bins) - min(mode_pd.bins)) / self.cent_ss
			pd.vals = np.concatenate((np.zeros(temp_left_shift), pd.vals))
		elif((min(pd.bins) - min(mode_pd.bins)) < 0):
			mode_left_shift = (min(mode_pd.bins) - min(pd.bins)) / self.cent_ss
			mode_pd.vals = np.concatenate((np.zeros(mode_left_shift), mode_pd.vals))

		### Alignment of the right end-points
		if((max(pd.bins) - max(mode_pd.bins)) > 0):
			mode_right_shift = (max(pd.bins) - max(mode_pd.bins)) / self.cent_ss
			mode_pd.vals = np.concatenate((mode_pd.vals, np.zeros(mode_right_shift)))
		elif((max(mode_pd.bins) - max(pd.bins)) > 0):    
			temp_right_shift = (max(mode_pd.bins) - max(pd.bins)) / self.cent_ss
   			pd.vals = np.concatenate((pd.vals, (np.zeros(temp_right_shift))))

   		return pd, mode_pd