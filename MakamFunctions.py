# -*- coding: utf-8 -*-
from scipy import stats
import pylab as pl
import numpy as np
import math

import PitchDistribution as p_d

def generate_pd(pitch_track, ref_freq=440, smooth_factor=7.5, cent_ss=7.5):		
	### Some extra interval is added to the beginning and end since the 
	### superposed Gaussian for smoothing would vanish after 3 sigmas.
	### The limits are also quantized to be a multiple of chosen step-size
	smoothening = (smooth_factor * np.sqrt(1/np.cov(pitch_track)))
	min_bin = (min(pitch_track) - (min(pitch_track) % smooth_factor)) - (5 * smooth_factor)  
	max_bin = (max(pitch_track) + (smooth_factor - (max(pitch_track) % smooth_factor))) + (5 * smooth_factor)

	pd_bins = np.arange(min_bin, max_bin, cent_ss)
	kde = stats.gaussian_kde(pitch_track, bw_method=smoothening)
	pd_vals = kde.evaluate(pd_bins)
	return p_d.PitchDistribution(pd_bins, pd_vals, kernel_width=smoothening, ref_freq=ref_freq), kde

def generate_pcd(pd):
	### Initializations
	pd_vals = pd.vals
	pd_bins = pd.bins
	step_size = pd.step_size
	pcd_bins = np.arange(0, 1200, step_size)
	pcd_vals = np.zeros(len(pcd_bins))

	###Octave wrapping
	for k in range(len(pd_bins)):
		idx = int((pd_bins[k] % 1200) / step_size)
		pcd_vals[idx] = pcd_vals[idx] + pd_vals[k]
		
	return p_d.PitchDistribution(pcd_bins, pcd_vals, kernel_width=pd.kernel_width, ref_freq = pd.ref_freq)

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