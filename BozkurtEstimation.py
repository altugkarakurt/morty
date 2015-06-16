# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as pl
from scipy.spatial import distance

import ModeFunctions as mf
import PitchDistribution as p_d

class BozkurtEstimation:

	def __init__(self, cent_ss=7.5, smooth_factor=7.5):
		self.smooth_factor = smooth_factor
		self.cent_ss = cent_ss

	def train(self, mode_name, txt_list, ref_freq_list, metric='pcd'):
		"""---------------------------------------------------------------------------------------
		This function handles everything related to supervised learning portion of this system. 
		It expects the list of text files containing the pitch tracks of the dataset, the array
		of their known tonics and generates the joint distribution of the mode and saves it.
		---------------------------------------------------------------------------------------"""
		mode_track = []
		for idx in range(len(txt_list)):
			cur_track = np.loadtxt(txt_list[idx])
			cur_cent_track = mf.hz_to_cent(cur_track, ref_freq=ref_freq_list[idx])
			for i in cur_cent_track:
				mode_track.append(i)
		joint_pd = mf.generate_pd(mode_track, smooth_factor=self.smooth_factor, cent_ss=self.cent_ss)[0]
		if(metric=='pcd'):
			joint_pcd = mf.generate_pcd(joint_pd)
			joint_pcd.save((mode_name + '_pcd.json'))
			return joint_pcd
		elif(metric=='pd'):
			joint_pd.save((mode_name + '_pd.json'))
			return joint_pd

	def estimate(self, pitch_track, mode_names=[], mode_name='', est_tonic=True, est_mode=True, mode_dir='./', distance_method="euclidean", metric='pcd', ref_freq=440):
		"""---------------------------------------------------------------------------------------
		This is the high level function that users are expected to interact with, for estimation
		purposes. Using the est_* flags, it is possible to estimate tonic, mode or both.
		---------------------------------------------------------------------------------------"""
		### The preliminaries to the *_estimate functions to be called later.
		cent_track = mf.hz_to_cent(pitch_track, ref_freq=ref_freq)
		dist = mf.generate_pd(cent_track, ref_freq=ref_freq, smooth_factor=self.smooth_factor, cent_ss=self.cent_ss)[0]
		if(metric=='pcd'):
			dist = mf.generate_pcd(dist)

		if(est_tonic and est_mode): # Joint estimation of tonic and mode
			### TODO
			return 0

		elif(est_tonic): # Estimate tonic only, mode is known.
			mode_dist = p_d.load((mode_name + '_' + metric + '.json'), mode_dir)
			if(metric=='pcd'):

				### Shifting to the global minima to prevent wrong detection of peaks
				shift_factor = dist.vals.tolist().index(min(dist.vals))
				dist = dist.shift(shift_factor)
				anti_freq = mf.cent_to_hz([dist.bins[shift_factor]], ref_freq=ref_freq)[0]
				peak_idxs, peak_vals = dist.detect_peaks()
				distance_vector = self.tonic_estimate(dist, mode_dist, peak_idxs, distance_method="euclidean", metric='pcd')

				### The detected tonic is converted back to hertz from cent
				return mf.cent_to_hz([dist.bins[peak_idxs[np.argmin(distance_vector)]]], anti_freq)[0]

			elif(metric=='pd'):
				peak_idxs, peak_vals = dist.detect_peaks()
   				origin =  np.where(dist.bins==0)[0][0]
				shift_idxs = [(idx - origin) for idx in peak_idxs]
				
				distance_vector = self.tonic_estimate(dist, mode_dist, shift_idxs, distance_method="euclidean", metric='pd')

				### The detected tonic is converted back to hertz from cent
				return mf.cent_to_hz([shift_idxs[np.argmin(distance_vector)] * self.cent_ss], ref_freq)[0]

		elif(est_mode): # Estimate mode only, tonic is known.
					
			### The candidate mode distributions are retrieved
			mode_dists = []
			for m in range(len(mode_names)):
				mode_dists.append(p_d.load(mode_names[m] + '_' + metric + '.json', mode_dir))
			distance_vector = self.mode_estimate(dist, mode_dists, ref_freq, distance_method='euclidean', metric=metric)
			return mode_names[np.argmin(distance_vector)]
		
		else:
			# Nothing is expected to be estimated
			return 0

	def tonic_estimate(self, dist, mode_dist, peak_idxs, distance_method="euclidean", metric='pcd'):
		"""---------------------------------------------------------------------------------------
		Given the mode (or candidate mode), compares the piece's distribution using the candidate
		tonics and returns the resultant distance vector to higher level functions.
		---------------------------------------------------------------------------------------"""
		### Mode is known, tonic is estimated.
		### Piece's distributon is generated
		
		if(metric=='pcd'):

			### Comparison of the piece's pcd with known mode's joint pcd
			return np.array(self.generate_distance_matrix(dist, peak_idxs, [mode_dist], method=distance_method))[:,0]

		elif(metric=='pd'):

			temp = p_d.PitchDistribution(dist.bins, dist.vals, kernel_width=dist.kernel_width, ref_freq=dist.ref_freq)
			temp, mode_dist = self.pd_zero_pad(temp, mode_dist)

    		### Filling both sides of vals with zeros, to make sure that the shifts won't drop any non-zero values
    		temp.vals = np.concatenate((np.zeros(abs(max(peak_idxs))), temp.vals, np.zeros(abs(min(peak_idxs)))))
    		mode_dist.vals = np.concatenate((np.zeros(abs(max(peak_idxs))), mode_dist.vals, np.zeros(abs(min(peak_idxs)))))
    		return np.array(self.generate_distance_matrix(temp, peak_idxs, [mode_dist], method=distance_method))[:,0]

	def mode_estimate(self, dist, mode_dists, tonic, distance_method='euclidean', metric='pcd'):
		"""---------------------------------------------------------------------------------------
		Given the tonic (or candidate tonic), compares the piece's distribution using the candidate
		modes and returns the resultant distance vector to higher level functions.
		---------------------------------------------------------------------------------------"""
		### Tonic is known, mode is estimated.

		if(metric=='pcd'):	
			return np.array(self.generate_distance_matrix(dist, [0], mode_dists, method=distance_method))

		elif(metric=='pd'):
			distance_vector = np.zeros(len(mode_dists))
			for i in range(len(mode_dists)):
				trial, mode_trial = self.pd_zero_pad(dist, mode_dists[i])
				distance_vector[i] = self.distance(trial, mode_trial, method=distance_method)
			return distance_vector

	def generate_distance_matrix(self, dist, peak_idxs, mode_dists, method='euclidean'):
		"""---------------------------------------------------------------------------------------
		Iteratively calculates the distance for all candidate tonics and candidate modes of a piece.
		The pair of candidates that give rise to the minimum value in this matrix is chosen as the
		estimate by the higher level functions.
		---------------------------------------------------------------------------------------"""
		result = np.zeros((len(peak_idxs), len(mode_dists)))
		for i in range(len(peak_idxs)):
			trial = dist.shift(peak_idxs[i])
			for j in range(len(mode_dists)):
				result[i][j] = self.distance(trial, mode_dists[j], method=method)
		return np.array(result)

	def distance(self, piece, trained, method='euclidean'):
		"""---------------------------------------------------------------------------------------
		Calculates the distance metric between two pitch distributions. This is called from
		estimation functions.
		---------------------------------------------------------------------------------------"""
		if(method=='euclidean'): # Euclidean Distance
			self.minkowski_distance(2, piece, trained)

		elif(method=='manhattan'): # Manhattan Distance
			self.minkowski_distance(1, piece, trained)

		elif(method=='l3'): # L3 Distance
			self.minkowski_distance(3, piece, trained)
			
		elif(method=='bhat'): # Bhattacharyya Distance
			d = 0
			for i in range(len(piece.vals)):
				d += math.sqrt(piece.vals[i] * trained.vals[i]);
			return (-math.log(d));

		else:
			return 0

	def minkowski_distance(self, degree, piece, trained):
		"""---------------------------------------------------------------------------------------
		Generic implementation of Minkowski distance. 
		When degree=1: This is Manhattan/City Blocks Distance
		When degree=2: This is Euclidean Distance
		When degree=3: This is L3 Distance
		---------------------------------------------------------------------------------------"""
		degree = degree * 1.0
		if(degree == 2.0):
			return distance.euclidean(piece.vals, trained.vals)
		else:
			d = 0
			for i in range(len(piece.vals)):
				d += ((abs(piece.vals[i] - trained.vals[i])) ** degree)
			return (d ** (1/degree))

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