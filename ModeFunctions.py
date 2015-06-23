# -*- coding: utf-8 -*-
import scipy as sp
import pylab as pl
import numpy as np
import math

from scipy import stats
from scipy.spatial import distance

import PitchDistribution as p_d

def load_track(txtname, txt_dir):
	"""---------------------------------------------------------------------------------------
	Loads the pitch track from a text file. The format for the examples is, such that, 0th
	column is time-stamps and 1st column is the corresponding frequency values.
	---------------------------------------------------------------------------------------"""
	return np.loadtxt(txt_dir + txtname + '.txt')

def generate_pd(pitch_track, ref_freq=440, smooth_factor=7.5, cent_ss=7.5, source='', segment='all'):		
	### Some extra interval is added to the beginning and end since the 
	### superposed Gaussian for smoothing would vanish after 3 sigmas.
	### The limits are also quantized to be a multiple of chosen step-size
	smoothening = (smooth_factor * np.sqrt(1/np.cov(pitch_track)))
	min_bin = (min(pitch_track) - (min(pitch_track) % smooth_factor)) - (5 * smooth_factor)  
	max_bin = (max(pitch_track) + (smooth_factor - (max(pitch_track) % smooth_factor))) + (5 * smooth_factor)

	pd_bins = np.arange(min_bin, max_bin, cent_ss)
	kde = stats.gaussian_kde(pitch_track, bw_method=smoothening)
	pd_vals = kde.evaluate(pd_bins)
	return p_d.PitchDistribution(pd_bins, pd_vals, kernel_width=smooth_factor, source=source, ref_freq=ref_freq, segment=segment)

def generate_pcd(pd):
	### Initializations
	pcd_bins = np.arange(0, 1200, pd.step_size)
	pcd_vals = np.zeros(len(pcd_bins))

	###Octave wrapping
	for k in range(len(pd.bins)):
		idx = int((pd.bins[k] % 1200) / pd.step_size)
		pcd_vals[idx] = pcd_vals[idx] + pd.vals[k]
		
	return p_d.PitchDistribution(pcd_bins, pcd_vals, kernel_width=pd.kernel_width, source=pd.source, ref_freq=pd.ref_freq, segment=pd.segmentation)

def hz_to_cent(hertz_track, ref_freq):
	### Hertz-to-Cent Conversion. Since the log of zeros are non_defined,
	### they're filtered out from the pitch track first.
	filtered_track = [freq for freq in hertz_track if freq > 0]
	cent_track = []
	for freq in filtered_track:
		cent_track.append(math.log((freq/ref_freq), 2.0) * 1200.0)
	return cent_track
	
def cent_to_hz(cent_track, ref_freq):
	hertz_track = []
	for cent in cent_track:
		hertz_track.append((2 ** (cent / 1200.0)) * ref_freq)
	return hertz_track

def generate_distance_matrix(dist, peak_idxs, mode_dists, method='euclidean'):
	"""---------------------------------------------------------------------------------------
	Iteratively calculates the distance for all candidate tonics and candidate modes of a piece.
	The pair of candidates that give rise to the minimum value in this matrix is chosen as the
	estimate by the higher level functions.
	---------------------------------------------------------------------------------------"""
	result = np.zeros((len(peak_idxs), len(mode_dists)))
	for i in range(len(peak_idxs)): 
		trial = dist.shift(peak_idxs[i])
		for j in range(len(mode_dists)):
			result[i][j] = distance(trial, mode_dists[j], method=method)
	return np.array(result)

def distance(piece, trained, method='euclidean'):
	"""---------------------------------------------------------------------------------------
	Calculates the distance metric between two pitch distributions. This is called from
	estimation functions.
	---------------------------------------------------------------------------------------"""
	if(method=='euclidean'):
		return minkowski_distance(2, piece, trained)

	elif(method=='manhattan'):
		return minkowski_distance(1, piece, trained)

	elif(method=='l3'):
		return minkowski_distance(3, piece, trained)
			
	elif(method=='bhat'):
		d = 0
		for i in range(len(piece.vals)):
			d += math.sqrt(piece.vals[i] * trained.vals[i]);
		return (-math.log(d));

	else:
		return 0

def minkowski_distance(degree, piece, trained):
	"""---------------------------------------------------------------------------------------
	Generic implementation of Minkowski distance. 
	When degree=1: This is Manhattan/City Blocks Distance
	When degree=2: This is Euclidean Distance
	When degree=3: This is L3 Distance
	---------------------------------------------------------------------------------------"""
	degree = degree * 1.0
	if(degree == 2.0):
		return sp.spatial.distance.euclidean(piece.vals, trained.vals)
	else:
		d = 0
		for i in range(len(piece.vals)):
			d += ((abs(piece.vals[i] - trained.vals[i])) ** degree)
		return (d ** (1/degree))

def pd_zero_pad(pd, mode_pd, cent_ss=7.5):
	"""---------------------------------------------------------------------------------------
	This function is only used while detecting tonic and working with pd as metric. It pads
	zeros from both sides of the values array to avoid losing non-zero values when comparing
	and to make sure the two PDs are of the same length 
	---------------------------------------------------------------------------------------"""
	### Alignment of the left end-points
	if((min(pd.bins) - min(mode_pd.bins)) > 0):
		temp_left_shift = (min(pd.bins) - min(mode_pd.bins)) / cent_ss
		pd.vals = np.concatenate((np.zeros(temp_left_shift), pd.vals))
	elif((min(pd.bins) - min(mode_pd.bins)) < 0):
		mode_left_shift = (min(mode_pd.bins) - min(pd.bins)) / cent_ss
		mode_pd.vals = np.concatenate((np.zeros(mode_left_shift), mode_pd.vals))

	### Alignment of the right end-points
	if((max(pd.bins) - max(mode_pd.bins)) > 0):
		mode_right_shift = (max(pd.bins) - max(mode_pd.bins)) / cent_ss
		mode_pd.vals = np.concatenate((mode_pd.vals, np.zeros(mode_right_shift)))
	elif((max(mode_pd.bins) - max(pd.bins)) > 0):    
		temp_right_shift = (max(mode_pd.bins) - max(pd.bins)) / cent_ss
		pd.vals = np.concatenate((pd.vals, (np.zeros(temp_right_shift))))

	return pd, mode_pd