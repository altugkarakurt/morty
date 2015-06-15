# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as pl
from scipy.spatial import distance

import MakamFunctions as mf
import PitchDistribution as p_d

class BozkurtEstimation:

	def __init__(self, cent_ss=7.5, smooth_factor=7.5):
		self.smooth_factor = smooth_factor
		self.cent_ss = cent_ss

	def train(self, makam, txt_list, ref_freq_list, metric='pcd'):
		makam_track = []
		for idx in range(len(txt_list)):
			cur_track = np.loadtxt(txt_list[idx])
			cur_cent_track = mf.hz_to_cent(cur_track, ref_freq=ref_freq_list[idx])
			for i in cur_cent_track:
				makam_track.append(i)
		joint_pd = mf.generate_pd(makam_track, smooth_factor=self.smooth_factor, cent_ss=self.cent_ss)[0]
		if(metric=='pcd'):
			joint_pcd = mf.generate_pcd(joint_pd)
			joint_pcd.save((makam + '_pcd.json'))
			return joint_pcd
		elif(metric=='pd'):
			joint_pd.save((makam + '_pd.json'))
			return joint_pd

	def tonic_estimate(self, pitch_track, makam, distance_method="euclidean", metric='pcd', dummy_freq=440):
		### Makam is known, tonic is estimated.
		### Piece's distributon is generated
		cent_track = mf.hz_to_cent(pitch_track, ref_freq=dummy_freq)
		dist = mf.generate_pd(cent_track, ref_freq=dummy_freq, smooth_factor=self.smooth_factor, cent_ss=self.cent_ss)[0]
		
		if(metric=='pcd'):
			dist = mf.generate_pcd(dist)

			### Shifting to the global minima to prevent wrong detection of peaks
			shift_factor = dist.vals.tolist().index(min(dist.vals))
			dist = dist.shift(shift_factor)
			anti_freq = mf.cent_to_hz([dist.bins[shift_factor]], ref_freq=dummy_freq)[0]
			peak_idxs, peak_vals = dist.detect_peaks()

			### Comparison of the piece's pcd with known makam's joint pcd
			makam_pcd = p_d.load((makam + '_pcd.json'))
			distance_vector = np.array(self.generate_distance_matrix(dist, peak_idxs, [makam_pcd], method=distance_method))[:,0]

			### The detected tonic is converted back to hertz from cent
			return mf.cent_to_hz([dist.bins[peak_idxs[np.argmin(distance_vector)]]], anti_freq)[0]

		elif(metric=='pd'):
			peak_idxs, peak_vals = dist.detect_peaks()
			origin =  np.where(dist.bins==0)[0][0]
			shift_idxs = [(idx - origin) for idx in peak_idxs]
			temp = p_d.PitchDistribution(dist.bins, dist.vals, kernel_width=dist.kernel_width, ref_freq=dist.ref_freq)
			makam_pd = p_d.load((makam + '_pd.json'))

			temp, makam_pd = self.pd_zero_pad(temp, makam_pd)

    		### Filling both sides of vals with zeros, to make sure that the shifts won't drop any non-zero values
    		temp.vals = np.concatenate((np.zeros(abs(max(shift_idxs))), temp.vals, np.zeros(abs(min(shift_idxs)))))
    		makam_pd.vals = np.concatenate((np.zeros(abs(max(shift_idxs))), makam_pd.vals, np.zeros(abs(min(shift_idxs)))))
    		
    		distance_vector = np.array(self.generate_distance_matrix(temp, shift_idxs, [makam_pd], method=distance_method))[:,0]

    		### The detected tonic is converted back to hertz from cent
    		return mf.cent_to_hz([shift_idxs[np.argmin(distance_vector)] * self.cent_ss], dummy_freq)[0]
	
	def makam_estimate(self, pitch_track, makams, tonic, distance_method='euclidean', metric='pcd'):
		### Tonic is known, makam is estimated.
		### Piece's distribution is generated
		cent_track = mf.hz_to_cent(pitch_track, ref_freq=tonic)
		dist = mf.generate_pd(cent_track, ref_freq=tonic, smooth_factor=self.smooth_factor, cent_ss=self.cent_ss)[0]
		
		### The candidate makam distributions are retrieved
		makam_dists = []
		for m in range(len(makams)):
			makam_dists.append(p_d.load(makams[m]))

		if(metric=='pcd'):
			dist = mf.generate_pcd(dist)
			
			### Distance vector generation
			distance_vector = np.array(self.generate_distance_matrix(dist, [0], makam_dists, method=distance_method))

		elif(metric=='pd'):
			distance_vector = np.zeros(len(makam_dists))
			for i in range(len(makam_dists)):
				trial, makam_trial = self.pd_zero_pad(dist, makam_dists[i])
				distance_vector[i] = self.distance(trial, makam_trial, method=distance_method)
		
		return makams[np.argmin(distance_vector)]

	def generate_distance_matrix(self, dist, peak_idxs, makam_dists, method='euclidean'):
		result = np.zeros((len(peak_idxs), len(makam_dists)))
		for i in range(len(peak_idxs)):
			trial = dist.shift(peak_idxs[i])
			for j in range(len(makam_dists)):
				result[i][j] = self.distance(trial, makam_dists[j], method=method)
		return np.array(result)

	def distance(self, piece, trained, method='euclidan'):

		if(method=='euclidean'):
			self.minkowski_distance(2, piece, trained)

		elif(method=='manhattan'):
			self.minkowski_distance(1, piece, trained)

		elif(method=='l3'):
			self.minkowski_distance(3, piece, trained)
			
		elif(method=='bhat'):
			d = 0
			for i in range(len(piece.vals)):
				d += math.sqrt(piece.vals[i] * trained.vals[i]);
			return (-math.log(d));

		else:
			return 0

	def minkowski_distance(self, degree, piece, trained):
		degree = degree * 1.0
		if(degree == 2.0):
			return distance.euclidean(piece.vals, trained.vals)
		else:
			d = 0
			for i in range(len(piece.vals)):
				d += ((abs(piece.vals[i] - trained.vals[i])) ** degree)
			return (d ** (1/degree))

	def pd_zero_pad(self, pd, makam_pd):
		"""---------------------------------------------------------------------------------------
		This function is only used when working with pd as metric. It pads zeros from both sides 
		of the values array to avoid losing non-zero values when comparing and to make sure the
		two pd's are of the same length 
		---------------------------------------------------------------------------------------"""
		### Alignment of the left end-points
		if((min(pd.bins) - min(makam_pd.bins)) > 0):
			temp_left_shift = (min(pd.bins) - min(makam_pd.bins)) / self.cent_ss
			pd.vals = np.concatenate((np.zeros(temp_left_shift), pd.vals))
		elif((min(pd.bins) - min(makam_pd.bins)) < 0):
			makam_left_shift = (min(makam_pd.bins) - min(pd.bins)) / self.cent_ss
			makam_pd.vals = np.concatenate((np.zeros(makam_left_shift), makam_pd.vals))

		### Alignment of the right end-points
		if((max(pd.bins) - max(makam_pd.bins)) > 0):
			makam_right_shift = (max(pd.bins) - max(makam_pd.bins)) / self.cent_ss
			makam_pd.vals = np.concatenate((makam_pd.vals, np.zeros(makam_right_shift)))
		elif((max(makam_pd.bins) - max(pd.bins)) > 0):    
			temp_right_shift = (max(makam_pd.bins) - max(pd.bins)) / self.cent_ss
   			pd.vals = np.concatenate((pd.vals, (np.zeros(temp_right_shift))))

   		return pd, makam_pd