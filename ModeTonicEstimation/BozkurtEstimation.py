# -*- coding: utf-8 -*-
import numpy as np
import os
from ModeTonicEstimation import ModeFunctions as mf
from ModeTonicEstimation import PitchDistribution as pd
import extras.fileOperations as fo


class BozkurtEstimation:
	"""-------------------------------------------------------------------------
	This is an implementation of the method proposed for tonic and makam
	estimation, in the following sources. This also includes some improvements
	to the method, such as the option of PCD along with PD, or the option of
	smoothing along with fine-grained pitch distributions. There is also the 
	option to get the first chunk of the input recording of desired length
	and only consider that portion for both estimation and training. 

	* A. C. Gedik, B.Bozkurt, 2010, "Pitch Frequency Histogram Based Music
	Information Retrieval for Turkish Music", Signal Processing, vol.10,
	pp.1049-1063.

	* B. Bozkurt, 2008, "An automatic pitch analysis method for Turkish maqam
	music", Journal of New Music Research 37 1–13.

	We require a set of recordings with annotated modes and tonics to train the
	mode models. Then, the unknown mode and/or tonic of an input recording is
	estimated by comparing it to these models.

	There are two functions and as their names suggest, one handles the training
	tasks and the other does the estimation once the trainings are completed.
	-------------------------------------------------------------------------"""

	def __init__(self, step_size=7.5, smooth_factor=7.5, chunk_size=0, frame_rate=128.0 / 44110):
		"""------------------------------------------------------------------------
		These attributes are wrapped as an object since these are used in both 
		training and estimation stages and must be consistent in both processes.
		---------------------------------------------------------------------------
		step_size       : Step size of the distribution bins
		smooth_factor : Std. deviation of the gaussian kernel used to smoothen the
						distributions. For further details, see generate_pd() of
						ModeFunctions.
		chunk_size    : The size of the recording to be considered. If zero, the
						entire recording will be used to generate the pitch
						distributions. If this is t, then only the first t seconds
						of the recording is used only and remaining is discarded.
		frame_rate     : The frame rate of the pitch tracks. Default is
						(128 = hopSize of the pitch extractor in pycompmusic)
						divided by 44100 audio sampling frequency. This is used
						to slice the pitch tracks according to the given chunk_size.
		------------------------------------------------------------------------"""
		self.smooth_factor = smooth_factor
		self.step_size = step_size
		self.chunk_size = chunk_size
		self.frame_rate = frame_rate

	def train(self, mode_name, pt_files, tonic_freqs, metric='pcd', save_dir=''):
		"""-------------------------------------------------------------------------
		For the mode trainings, the requirements are a set of recordings with 
		annotated tonics for each mode under consideration. This function only
		expects the recordings' pitch tracks and corresponding tonics as lists.
		The two lists should be indexed in parallel, so the tonic of ith pitch
		track in the pitch track list should be the ith element of tonic list.
		Once training is completed for a mode, the model wouldbe generated as a 
		PitchDistribution object and saved in a JSON file. For loading these objects
		and other relevant information about the data structure, see the
		PitchDistribution class.
		----------------------------------------------------------------------------
		mode_name     : Name of the mode to be trained. This is only used for naming
						the resultant JSON file, in the form "mode_name.json"
		pt_files      : List of files with pitch tracks extracted from the recording
						(i.e. single-column files with frequencies)
		tonic_freqs   : List of annotated tonic frequencies of recordings
		metric        : Whether the model should be octave wrapped (Pitch Class
						Distribution: PCD) or not (Pitch Distribution: PD)
		save_dir      : Where to save the resultant JSON files.
		-------------------------------------------------------------------------"""

		# To generate the model pitch distribution of a mode, pitch track of each
		# recording is iteratively converted to cents, according to their respective
		# annotated tonics. Then, these are appended to mode_track and a very long
		# pitch track is generated, as if it is a single very long recording. The 
		# pitch distribution of this track is the mode's model distribution.

		# Normalize the pitch tracks of the mode wrt the tonic frequency and concatenate
		for pf, tonic in zip(pt_files, tonic_freqs):
			pitch_track = np.loadtxt(pf)
			if pitch_track.ndim > 1:  # assume the first col is time, the second is pitch and the rest is labels etc
				pitch_track = pitch_track[:,1]

			if self.chunk_size == 0:  # use the complete pitch track
				mode_track = mf.hz_to_cent(pitch_track, ref_freq=tonic)
			else:  # slice and used the start of the pitch track
				time_track = np.arange(0, self.frame_rate * len(pitch_track), self.frame_rate)
				pitch_track, segs = mf.slice(time_track, pitch_track, mode_name, self.chunk_size)
				mode_track = mf.hz_to_cent(pitch_track[0], ref_freq=tonic)

		seglen = 'all' if self.chunk_size == 0 else (segs[0][1], segs[0][2])

		# generate the pitch distribution
		pitch_distrib = mf.generate_pd(mode_track, smooth_factor=self.smooth_factor,
		                               step_size=self.step_size, source=pt_files, segment=seglen)
		if metric == 'pcd':  # convert to pitch class distribution, if specified
			pitch_distrib = mf.generate_pcd(pitch_distrib)

		# save the model to a file, if requested
		if save_dir:
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)
			pitch_distrib.save(mode_name + '.json', save_dir=save_dir)

		return pitch_distrib

	def estimate(self, pitch_track, mode_in='./', tonic_freq=None, rank=1,
	             distance_method="bhat", metric='pcd'):
		"""-------------------------------------------------------------------------
		This is the ultimate estimation function. There are three different types
		of estimations.

		1) Joint Estimation: Neither the tonic nor the mode of the recording is known.
		Then, joint estimation estimates both of these parameters without any prior
		knowledge about the recording.
		To use this: est_mode and est_tonic flags should be True since both are to
		be estimated. In this case tonic_freq and mode_name parameters are not used,
		since these are used to pass the annotated data about the recording.

		2) Tonic Estimation: The mode of the recording is known and tonic is to be
		estimated. This is generally the most accurate estimation among the three.
		To use this: est_tonic should be True and est_mode should be False. In this
		case tonic_freq  and mode_names parameters are not used since tonic isn't
		known a priori and mode is known and hence there is no candidate mode.

		3) Mode Estimation: The tonic of the recording is known and mode is to be
		estimated.
		To use this: est_mode should be True and est_tonic should be False. In this
		case mode_name parameter isn't used since the mode annotation is not
		available. It can be ignored.
		----------------------------------------------------------------------------
		pitch_track     : Pitch track of the input recording whose tonic and/or mode
						is to be estimated. This is only a 1-D list of frequency
						values.
		mode_in         : The mode input, If it is a filename or distribution object,
						the mode is treated as known and only tonic will be estimated.
						If a directory with the json files or dictionary of
						distributions (per mode) is given, the mode will be estimated.
						In case of directory, the modes will be taken as the json
						filenames.
		tonic_freq      : Annotated tonic of the recording. If it's unknown, we use
						an arbitrary value, so this can be ignored.
		rank            : The number of estimations expected from the system. If
						this is 1, estimation returns the most likely tonic, mode
						or tonic/mode pair. If it is n, it returns a sorted list
						of tuples of length n, each containing a tonic/mode pair. 
		distance_method : The choice of distance methods. See distance() in
						ModeFunctions for more information.
		metric          : Whether the model should be octave wrapped (Pitch Class
						Distribution: PCD) or not (Pitch Distribution: PD)
		-------------------------------------------------------------------------"""

		# parse mode input
		try:
			if os.path.isdir(mode_in): # folder
				mode_files = fo.getFileNamesInDir(mode_in)[2]  # directory is given
				est_mode = True  # do mode estimation
				mode_names = [os.path.splitext(m)[0] for m in mode_files]
				models = [pd.load(m) for m in mode_files]
			elif os.path.isfile(mode_in): # json file
				est_mode = False # mode already known
				model = pd.load(mode_in)
		except TypeError:
			try:  # dictionary of models
				if isinstance(mode_in, pd.PitchDistribution):  # mode is loaded
					est_mode = False  # mode already known
					model = mode_in
				elif isinstance(mode_in, dict):  # models of all modes are loaded
					est_mode = True  # do mode estimation
					mode_names = mode_in.keys()
					models = [mode_in[m] for m in mode_names]
			except:
				print "Unknown mode input!"

		# parse tonic input
		if tonic_freq:  # tonic is already known;
			est_tonic = False
		else:
			est_tonic = True
			tonic_freq = 440  # take A4 as the dummy frequency value for cent conversion; it doesnt affect anything

		if not (est_tonic or est_mode):
			print "Both tonic and mode are known!"
			return -1

		# slice the pitch track if specified
		if self.chunk_size > 0:
			time_track = np.arange(0, self.frame_rate * len(pitch_track), self.frame_rate)
			pitch_track, segs = mf.slice(time_track, pitch_track, '', self.chunk_size)

		# normalize pitch track according to the given tonic frequency
		cent_track = mf.hz_to_cent(pitch_track, ref_freq=tonic_freq)

		# Pitch distribution of the input recording is generated
		distrib = mf.generate_pd(cent_track, ref_freq=tonic_freq, smooth_factor=self.smooth_factor,
		                         step_size=self.step_size)

		# convert to PCD, if specified
		distrib = mf.generate_pcd(distrib) if (metric == 'pcd') else distrib

		# Saved mode models are loaded and output variables are initiated
		tonic_ranked = [('', 0) for x in range(rank)]
		mode_ranked = [('', 0) for x in range(rank)]

		# Preliminary steps for tonic identification
		if est_tonic:
			if metric == 'pcd':
				# If there happens to be a peak at the last (and first due to the circular
				# nature of PCD) sample, it is considered as two peaks, one at the end and
				# one at the beginning. To prevent this, we find the global minima (as it
				# is easy to compute) of the distribution and make it the new reference
				# frequency, i.e. shift it to the beginning.
				shift_factor = distrib.vals.tolist().index(min(distrib.vals))
				distrib = distrib.shift(shift_factor)

				# update to the new reference frequency after shift
				tonic_freq = mf.cent_to_hz([distrib.bins[shift_factor]], ref_freq=tonic_freq)[0]

				# Find the peaks of the distribution. These are the tonic candidates.
				peak_idxs, peak_vals = distrib.detect_peaks()
			elif metric == 'pd':
				# Find the peaks of the distribution. These are the tonic candidates
				peak_idxs, peak_vals = distrib.detect_peaks()

				# The number of samples to be shifted is the list [peak indices - zero bin]
				# origin is the bin with value zero and the shifting is done w.r.t. it.
				origin = np.where(distrib.bins == 0)[0][0]
				shift_idxs = [(idx - origin) for idx in peak_idxs]

		# Joint Estimation
		if (est_tonic and est_mode):
			if (metric == 'pd'):
				# Since PD lengths aren't equal, we zero-pad the distributions for comparison
				# tonic_estimate() of ModeFunctions just does that. It can handle only
				# a single column, so the columns of the matrix are iteratively generated
				dist_mat = np.zeros((len(shift_idxs), len(models)))
				for m, model in enumerate(models):
					dist_mat[:, m] = mf.tonic_estimate(distrib, shift_idxs, model, distance_method=distance_method,
					                                   metric=metric, step_size=self.step_size)
			elif (metric == 'pcd'):
				# PCD doesn't require any preliminary steps. Generate the distance matrix.
				# The rows are tonic candidates and columns are mode candidates.
				dist_mat = mf.generate_distance_matrix(distrib, peak_idxs, models, method=distance_method)

			# Distance matrix is ready now. For each rank, (or each pair of
			# tonic-mode estimate pair) the loop is iterated. When the first
			# best estimate is found it's changed to the worst, so in the
			# next iteration, the estimate would be the second best and so on.
			for r in range(min(rank, len(peak_idxs))):
				# The minima of the distance matrix is found. This is when the
				# distribution is the most similar to a mode distribution, according
				# to the corresponding tonic estimate. The corresponding tonic
				# and mode pair is our current estimate.
				min_row = np.where((dist_mat == np.amin(dist_mat)))[0][0]
				min_col = np.where((dist_mat == np.amin(dist_mat)))[1][0]
				# Due to the precaution step of PCD, the reference frequency is
				# changed. That's why it's treated differently than PD. Here,
				# the cent value of the tonic estimate is converted back to Hz.
				if (metric == 'pcd'):
					tonic_ranked[r] = (mf.cent_to_hz([distrib.bins[peak_idxs[min_row]]],
					                                 tonic_freq)[0], dist_mat[min_row][min_col])
				elif (metric == 'pd'):
					tonic_ranked[r] = (mf.cent_to_hz([shift_idxs[min_row] * self.step_size],
					                                 tonic_freq)[0], dist_mat[min_row][min_col])
				# Current mode estimate is recorded.
				mode_ranked[r] = (mode_names[min_col], dist_mat[min_row][min_col])
				# The minimum value is replaced with a value larger than maximum,
				# so we won't return this estimate pair twice.
				dist_mat[min_row][min_col] = (np.amax(dist_mat) + 1)
			return mode_ranked, tonic_ranked

		# Tonic Estimation
		elif (est_tonic):
			# This part assigns the special case changes to standard variables,
			# so that we can treat PD and PCD in the same way, as much as
			# possible. 
			peak_idxs = shift_idxs if (metric == 'pd') else peak_idxs
			tonic_freq = tonic_freq if (metric == 'pcd') else tonic_freq

			# Distance vector is generated. In the mode_estimate() function
			# of ModeFunctions, PD and PCD are treated differently and it
			# handles the special cases such as zero-padding. The mode is
			# already known, so there is only one model to be compared. Each
			# entry corresponds to one tonic candidate.
			distance_vector = mf.tonic_estimate(distrib, peak_idxs, model, distance_method=distance_method,
			                                    metric=metric, step_size=self.step_size)

			# Distance vector is ready now. For each rank, the loop is iterated.
			# When the first best estimate is found it's changed to be the worst,
			# so in the next iteration, the estimate would be the second best
			# and so on
			for r in range(min(rank, len(peak_idxs))):
				# Minima is found, corresponding tonic candidate is our current
				# tonic estimate
				idx = np.argmin(distance_vector)
				# Due to the changed reference frequency in PCD's precaution step,
				# PCD and PD are treated differently here. 
				# TODO: review here, this might be tedious due to 257th line.
				if (metric == 'pcd'):
					tonic_ranked[r] = (mf.cent_to_hz([distrib.bins[peak_idxs[idx]]],
					                                 tonic_freq)[0], distance_vector[idx])
				elif (metric == 'pd'):
					tonic_ranked[r] = (mf.cent_to_hz([shift_idxs[idx] * self.step_size],
					                                 tonic_freq)[0], distance_vector[idx])
				# Current minima is replaced with a value larger than maxima,
				# so that we won't return the same estimate twice.
				distance_vector[idx] = (np.amax(distance_vector) + 1)
			return tonic_ranked

		# Mode Estimation
		elif (est_mode):
			# Distance vector is generated. Again, mode_estimate() of
			# ModeFunctions handles the different approach required for
			# PCD and PD. Since tonic is known, the distributions aren't
			# shifted and are only compared to candidate mode models.
			distance_vector = mf.mode_estimate(distrib, models, distance_method=distance_method, metric=metric,
			                                   step_size=self.step_size)

			# Distance vector is ready now. For each rank, the loop is iterated.
			# When the first best estimate is found it's changed to be the worst,
			# so in the next iteration, the estimate would be the second best
			# and so on
			for r in range(min(rank, len(mode_names))):
				# Minima is found, corresponding mode candidate is our current
				# mode estimate
				idx = np.argmin(distance_vector)
				mode_ranked[r] = (mode_names[idx], distance_vector[idx])
				# Current minima is replaced with a value larger than maxima,
				# so that we won't return the same estimate twice.
				distance_vector[idx] = (np.amax(distance_vector) + 1)
			return mode_ranked

		else:
			# Nothing is expected to be estimated.
			return 0
