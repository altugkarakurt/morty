# -*- coding: utf-8 -*-
import scipy as sp
import numpy as np
import math

import os
from scipy import stats
from scipy.spatial import distance

import PitchDistribution as p_d


def load_track(txt_name, txt_dir):
	"""---------------------------------------------------------------------------------------
    Loads the pitch track from a text file. The format for the examples is, such that, 0th
    column is time-stamps and 1st column is the corresponding frequency values.
    ---------------------------------------------------------------------------------------"""
	return np.loadtxt(os.path.join(txt_dir, txt_name))


def generate_pd(cent_track, ref_freq=440, smooth_factor=7.5, cent_ss=7.5, source='', segment='all', overlap='-'):
	### Some extra interval is added to the beginning and end since the
	### superposed Gaussian for smoothing would vanish after 3 sigmas.
	### The limits are also quantized to be a multiple of chosen step-size
	### smooth_factor = standard deviation fo the gaussian kernel

	# filter out the Nan, -infinity and +infinity from the pitch track, if exists
	# TODO
	# get the pitch values
	if (smooth_factor > 0): # KDE
		print "pre generate pd"
		# convert the standard deviation of the Gaussian kernel to the bandwidth of the smoothening constant
		smoothening = (smooth_factor * np.sqrt(1 / np.cov(cent_track)))
		print "after smooth"
		# take the min/max longer such that the pitch values in the boundaries can decay
		min_bin = (min(cent_track) - (min(cent_track) % smooth_factor)) - (5 * smooth_factor)
		max_bin = (max(cent_track) + (smooth_factor - (max(cent_track) % smooth_factor))) + (5 * smooth_factor)
		print "after bin"
		# generate the pitch distribution bins; make sure it crosses 0
		pd_bins = np.concatenate([np.arange(0, min_bin, -cent_ss)[::-1], np.arange(cent_ss, max_bin, cent_ss)])
		print "after bin cat"
		# a rare case is when min_bin and max_bin are both greater than 0 in this case the first array will be empty
		# resulting in pd_bins in the range of cent_ss to max_bin. If it occurs we should put a 0 to the start of the
		# array
		pd_bins = pd_bins if 0 in pd_bins else np.insert(pd_bins, 0, 0)
		print "after bin oglu bin"
		# generate the kernel density estimate and evaluate at the given bins
		kde = stats.gaussian_kde(cent_track, bw_method=smoothening)
		pd_vals = kde.evaluate(pd_bins)
		print "after kde"
	else: #histogram
		# get the min and max possible values of the histogram edges; the actual values will be dependent on "cent_ss"
		min_edge = min(cent_track) - (cent_ss / 2.0)
		max_edge = max(cent_track) + (cent_ss / 2.0)

		# generate the pitch distribution bins; make 0 is the center of a bin
		pd_edges = np.concatenate([np.arange(-cent_ss/2.0, min_edge, -cent_ss)[::-1], np.arange(cent_ss/2.0, max_edge, cent_ss)])

		# a rare case is when min_bin and max_bin are both greater than 0 in this case the first array will be empty
		# resulting in pd_bins in the range of cent_ss to max_bin. If it occurs we should put a -cent_ss/2 to the start of the
		# array
		pd_edges = pd_edges if -cent_ss/2.0 in pd_edges else np.insert(pd_edges, 0, (-cent_ss/2.0))
		pd_edges = pd_edges if cent_ss/2.0 in pd_edges else np.append(pd_edges, (cent_ss/2.0))

		pd_vals, pd_edges = np.histogram(cent_track, bins=pd_edges, density=True)
		pd_bins = np.convolve(pd_edges, [0.5,0.5])[1:-1]

	if(len(pd_bins) != len(pd_vals)):
		raise ValueError('Lengths of bins and Vals are different')
	print "pre return"
	return p_d.PitchDistribution(pd_bins, pd_vals, kernel_width=smooth_factor, source=source, ref_freq=ref_freq,
	                             segment=segment, overlap=overlap)

def generate_pcd(pd):
	### Initializations
	pcd_bins = np.arange(0, 1200, pd.step_size)
	pcd_vals = np.zeros(len(pcd_bins))

	###Octave wrapping
	for k in range(len(pd.bins)):
		idx = int((pd.bins[k] % 1200) / pd.step_size)
		idx = idx if idx != 160 else 0
		pcd_vals[idx] = pcd_vals[idx] + pd.vals[k]
	
	#Due to the floating point issues in Python, the step_size of Pitch Distributions might
	#not be exactly equal to 7.5, but 7.4999... etc. In these cases 1200 cents is also
	#generated. Since we are working in a single octave, these are folded into 0 cents.


	return p_d.PitchDistribution(pcd_bins, pcd_vals, kernel_width=pd.kernel_width, source=pd.source,
	                             ref_freq=pd.ref_freq, segment=pd.segmentation, overlap=pd.overlap)


def hz_to_cent(hertz_track, ref_freq):
	### Hertz-to-Cent Conversion. Since the log of zeros are non_defined,
	### they're filtered out from the pitch track first.
	filtered_track = [freq for freq in hertz_track if freq > 0]
	cent_track = []
	for freq in filtered_track:
		cent_track.append(math.log((freq / ref_freq), 2.0) * 1200.0)
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
	if (method == 'euclidean'):
		return sp.spatial.distance.euclidean(piece.vals, trained.vals)

	elif (method == 'manhattan'):
		return sp.spatial.distance.minkowski(piece.vals, trained.vals, 1)

	elif (method == 'l3'):
		return sp.spatial.distance.minkowski(piece.vals, trained.vals, 3)

	elif (method == 'bhat'):
		d = 0
		for i in range(len(piece.vals)):
			d += math.sqrt(piece.vals[i] * trained.vals[i]);
		return (-math.log(d))

	elif (method == 'intersection'): #to be renamed as inverse intersection
		d = 0
		for j in range(len(piece.vals)):
			d += min(piece.vals[j], trained.vals[j])
		return (len(piece.vals)) / d

	elif (method == 'corr'):
		return 1.0 - np.correlate(piece.vals, trained.vals)

	else:
		return 0


def pd_zero_pad(pd, mode_pd, cent_ss=7.5):
	"""---------------------------------------------------------------------------------------
	This function is only used while detecting tonic and working with pd as metric. It pads
	zeros from both sides of the values array to avoid losing non-zero values when comparing
	and to make sure the two PDs are of the same length
	---------------------------------------------------------------------------------------"""
	### In the following procedure, the padding process requires the two bin lists to have
	### an intersection. This is ensured by the generate_pd function.

	#find the number of missing bins in the left and right sides of pd 
	diff_bins = set(mode_pd.bins) - set(pd.bins)
	num_left_missing = len([x for x in diff_bins if x < min(pd.bins)]) 
	num_right_missing = len([x for x in diff_bins if x > max(pd.bins)])
	pd.vals = np.concatenate((np.zeros(num_left_missing), pd.vals, np.zeros(num_right_missing)))

	#find the number of missing bins in the left and right sides of mode_pd 
	#this code is identical to the previous block. TODO: make them modular later
	diff_bins = set(pd.bins) - set(mode_pd.bins)
	num_left_missing = len([x for x in diff_bins if x < min(mode_pd.bins)]) 
	num_right_missing = len([x for x in diff_bins if x > max(mode_pd.bins)])
	mode_pd.vals = np.concatenate((np.zeros(num_left_missing), mode_pd.vals, np.zeros(num_right_missing)))

	return pd, mode_pd


def tonic_estimate(dist, peak_idxs, mode_dist, distance_method="euclidean", metric='pcd', cent_ss=7.5):
	"""---------------------------------------------------------------------------------------
    Given the mode (or candidate mode), compares the piece's distribution using the candidate
    tonics and returns the resultant distance vector to higher level functions.
    ---------------------------------------------------------------------------------------"""
	### Mode is known, tonic is estimated.
	### Piece's distributon is generated

	if (metric == 'pcd'):
		return np.array(generate_distance_matrix(dist, peak_idxs, [mode_dist], method=distance_method))[:, 0]

	elif (metric == 'pd'):
		temp = p_d.PitchDistribution(dist.bins, dist.vals, kernel_width=dist.kernel_width, source=dist.source,
		                             ref_freq=dist.ref_freq, segment=dist.segmentation)
		temp, mode_dist = pd_zero_pad(temp, mode_dist, cent_ss=cent_ss)

		### Filling both sides of vals with zeros, to make sure that the shifts won't drop any non-zero values
		temp.vals = np.concatenate((np.zeros(abs(max(peak_idxs))), temp.vals, np.zeros(abs(min(peak_idxs)))))
		mode_dist.vals = np.concatenate((np.zeros(abs(max(peak_idxs))), mode_dist.vals, np.zeros(abs(min(peak_idxs)))))

		return np.array(generate_distance_matrix(temp, peak_idxs, [mode_dist], method=distance_method))[:, 0]


def mode_estimate(dist, mode_dists, distance_method='euclidean', metric='pcd', cent_ss=7.5):
	"""---------------------------------------------------------------------------------------
    Given the tonic (or candidate tonic), compares the piece's distribution using the candidate
    modes and returns the resultant distance vector to higher level functions.
    ---------------------------------------------------------------------------------------"""

	if (metric == 'pcd'):
		distance_vector = np.array(generate_distance_matrix(dist, [0], mode_dists, method=distance_method))[0]

	elif (metric == 'pd'):
		print "pre distance"

		distance_vector = np.zeros(len(mode_dists))
		for i in range(len(mode_dists)):
			trial = p_d.PitchDistribution(dist.bins, dist.vals, kernel_width=dist.kernel_width, source=dist.source,
			                              ref_freq=dist.ref_freq, segment=dist.segmentation)
			trial, mode_trial = pd_zero_pad(trial, mode_dists[i], cent_ss=cent_ss)
			distance_vector[i] = distance(trial, mode_trial, method=distance_method)
	return distance_vector


def slice(time_track, pitch_track, pt_source, chunk_size, threshold=0.5, overlap=0):
	segments = []
	seg_lims = []
	last = 0
	for k in np.arange(1, (int(max(time_track) / chunk_size) + 1)):
		cur = 1 + max(np.where(time_track < chunk_size * k)[0])
		segments.append(pitch_track[last:(cur - 1)])
		seg_lims.append((pt_source, int(round(time_track[last])),
		                 int(round(time_track[cur - 1]))))  # 0 - source, 1 - init, 2 - final
		last = 1 + max(np.where(time_track < chunk_size * k * (1 - overlap))[0]) if (overlap > 0) else cur
	if ((max(time_track) - time_track[last]) >= (chunk_size * threshold)):
		segments.append(pitch_track[last:])
		seg_lims.append((pt_source, int(round(time_track[last])), int(round(time_track[len(time_track) - 1]))))
	elif (last == 0):  # If the runtime of the track is below the threshold, keep it as it is
		segments.append(pitch_track)
		seg_lims.append((pt_source, 0, int(round(time_track[len(time_track) - 1]))))
	return segments, seg_lims
