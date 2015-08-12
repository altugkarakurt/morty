# -*- coding: utf-8 -*-
import numpy as np
import maths
import ModeFunctions as mf
import PitchDistribution as p_d

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

	def __init__(self, cent_ss=7.5, smooth_factor=7.5, chunk_size=0):
		"""------------------------------------------------------------------------
		These attributes are wrapped as an object since these are used in both 
		training and estimation stages and must be consistent in both processes.
		---------------------------------------------------------------------------
		cent_ss       : Step size of the distribution bins
		smooth_factor : Std. deviation of the gaussian kernel used to smoothen the
		                distributions. For further details, see generate_pd() of
		                ModeFunctions.
		chunk_size    : The size of the recording to be considered. If zero, the
		                entire recording will be used to generate the pitch
		                distributions. If this is t, then only the first t seconds
		                of the recording is used only and remaining is discarded.
		------------------------------------------------------------------------"""
		self.smooth_factor = smooth_factor
		self.cent_ss = cent_ss
		self.chunk_size = chunk_size

	def train(self, mode_name, pt_list, ref_freq_list, pt_dir='./', metric='pcd', save_dir='./'):
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
		pt_list       : List of pitch tracks (i.e. 1-D list of frequencies)
		ref_freq_list : List of annotated tonics of recordings
		pt_dir        : The directory where pitch tracks are stored.
		metric        : Whether the model should be octave wrapped (Pitch Class 
			            Distribution: PCD) or not (Pitch Distribution: PD)
		save_dir      : Where to save the resultant JSON files.
		-------------------------------------------------------------------------"""
		
		# To generate the model pitch distribution of a mode, pitch track of each
		# recording is iteratively converted to cents, according to their respective
		# annotated tonics. Then, these are appended to mode_track and a very long
		# pitch track is generated, as if it is a single very long recording. The 
		# pitch distribution of this track is the mode's model distribution.

		# This loop creates the joint pitch track of the mode in cents.
		mode_track = []
		for idx in range(len(pt_list)):

			# chunk_size == 0 means the recordings aren't going to be sliced and used
			# as they are.
			if (self.chunk_size == 0):
				cur_track = mf.load_track(pt_list[idx], pt_dir)[:,1]
				cur_cent_track = mf.hz_to_cent(cur_track, ref_freq=ref_freq_list[idx])
				joint_seg = 'all'

			# if slicing is desired, the pieces are sliced and only the first chunk
			# is used.
			else:
				tmp_track = mf.load_track(pt_list[idx], pt_dir)[:,1]
				time_track = mf.load_track(pt_list[idx], pt_dir)[:,0]
				cur_track, segs = mf.slice(time_track, tmp_track, mode_name, self.chunk_size)
				cur_cent_track = mf.hz_to_cent(cur_track[0], ref_freq=ref_freq_list[idx])
				joint_seg = (segs[0][1], segs[0][2])
			for i in cur_cent_track:
				mode_track.append(i)

		# Distribution of the joint pitch track is generated as the model.
		joint_dist = mf.generate_pd(mode_track, smooth_factor=self.smooth_factor,
			                        cent_ss=self.cent_ss, source=pt_list, segment=joint_seg)
		if(metric=='pcd'):
			joint_dist = mf.generate_pcd(joint_dist)

		# Mode model is saved.
		joint_dist.save((mode_name + '.json'), save_dir=save_dir)

	def estimate(self, pitch_track, time_track, mode_names=[], mode_name='', mode_dir='./',
		         est_tonic=True, est_mode=True, rank=1, distance_method="euclidean",
		         metric='pcd', ref_freq=440):
		"""-------------------------------------------------------------------------
		This is the ultimate estimation function. There are three different types
		of estimations.

		1) Joint Estimation: Neither the tonic nor the mode of the recording is known.
		Then, joint estimation estimates both of these parameters without any prior
		knowledge about the recording.
		To use this: est_mode and est_tonic flags should be True since both are to
		be estimated. In this case ref_freq and mode_name parameters are not used,
		since these are used to pass the annotated data about the recording.

		2) Tonic Estimation: The mode of the recording is known and tonic is to be
		estimated. This is generally the most accurate estimation among the three.
		To use this: est_tonic should be True and est_mode should be False. In this
		case ref_freq  and mode_names parameters are not used since tonic isn't
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
		time_track      : The timestamps of the pitch track. This is only used for
		                  slicing.
		mode_dir        : The directory where the mode models are stored. This is to
		                  load the annotated mode or the candidate mode.
		mode_names      : Names of the candidate modes. These are used when loading
		                  the mode models. If the mode isn't estimated, this parameter
		                  isn't used and can be ignored.
		mode_name       : Annotated mode of the recording. If it's not known and to be
		                  estimated, this parameter isn't used and can be ignored.
		est_tonic       : Whether tonic is to be estimated or not. If this flag is
		                  False, ref_freq is treated as the annotated tonic.
		est_mode        : Whether mode is to be estimated or not. If this flag is
		                  False, mode_name is treated as the annotated mode.
		rank            : The number of estimations expected from the system. If
		                  this is 1, estimation returns the most likely tonic, mode
		                  or tonic/mode pair. If it is n, it returns a sorted list
		                  of tuples of length n, each containing a tonic/mode pair. 
		distance_method : The choice of distance methods. See distance() in
		                  ModeFunctions for more information.
		metric          : Whether the model should be octave wrapped (Pitch Class
		                  Distribution: PCD) or not (Pitch Distribution: PD)
		ref_freq        : Th
		-------------------------------------------------------------------------"""

		# Preliminaries before the estimations
		# Pitch track of the input recording is (first sliced if necessary) converted
		# to cents.
		if (self.chunk_size > 0):
			cur_track, segs = mf.slice(time_track, pitch_track, mode_name, self.chunk_size)
			cent_track = mf.hz_to_cent(cur_track[0], ref_freq=ref_freq)
		else:
			cent_track = mf.hz_to_cent(pitch_track, ref_freq=ref_freq)
		# Pitch distribution of the input recording is generated
		dist = mf.generate_pd(cent_track, ref_freq=ref_freq,
			                  smooth_factor=self.smooth_factor, cent_ss=self.cent_ss)
		dist = mf.generate_pcd(dist) if (metric=='pcd') else dist
		# Saved mode models are loaded and output variables are initiated
		mode_dists = [(p_d.load((m + '.json'), mode_dir)) for m in mode_names]
		mode_dist = p_d.load((mode_name + '.json'), mode_dir) if (mode_name!='') else None
		tonic_list = [('',0) for x in range(rank)]
		mode_list = [('',0) for x in range(rank)]

		# If tonic will be estimated, there are certain common preliminary steps, 
		# regardless of the process being a joint estimation of a tonic estimation.
		if(est_tonic):
			if(metric=='pcd'):
				# This is a precaution step, just to be on the safe side. If there
				# happens to be a peak at the last (and first due to the circular nature
				# of PCD) sample, it is considered as two peaks, one at the end and
				# one at the beginning. To prevent this, we find the global minima
				# of the distribution and shift it to the beginning, i.e. make it the
				# new reference frequency. This new reference could have been any other
				# as long as there is no peak there, but minima is fairly easy to find.
				shift_factor = dist.vals.tolist().index(min(dist.vals))
				dist = dist.shift(shift_factor)
				# anti-freq is the new reference frequency after shift, as mentioned
				# above.
				anti_freq = mf.cent_to_hz([dist.bins[shift_factor]], ref_freq=ref_freq)[0]
				# Peaks of the distribution are found and recorded. These will be treated
				# as tonic candidates.
				peak_idxs, peak_vals = dist.detect_peaks()
			elif(metric=='pd'):
				# Since PD isn't circular, the precaution in PCD is unnecessary here.
				# Peaks of the distribution are found and recorded. These will be treated
				# as tonic candidates.
				peak_idxs, peak_vals = dist.detect_peaks()
				# The number of samples to be shifted is the list [peak indices - zero bin]
				# origin is the bin with value zero and the shifting is done w.r.t. it.
				origin =  np.where(dist.bins==0)[0][0]
				shift_idxs = [(idx - origin) for idx in peak_idxs]

		# Here the actual estimation steps begin

		#Joint Estimation
		if(est_tonic and est_mode):
			if(metric=='pcd'):
				# PCD doesn't require any prelimimary steps. Generates the distance matrix.
				# The rows are tonic candidates and columns are mode candidates.
				dist_mat = mf.generate_distance_matrix(dist, peak_idxs, mode_dists, 
					                                   method=distance_method)

			elif(metric=='pd'):
				# Since PD lengths aren't equal, zero padding is required and
				# tonic_estimate() of ModeFunctions just does that. It can handle only
				# a single column, so the columns of the matrix are iteratively generated
				dist_mat = np.zeros((len(shift_idxs), len(mode_dists)))
				for m in range(len(mode_dists)):
					dist_mat[:,m] = mf.tonic_estimate(dist, shift_idxs, mode_dists[m], 
						                              distance_method=distance_method,
						                              metric=metric, cent_ss=self.cent_ss)
			
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
				if(metric=='pcd'):
					tonic_list[r] = (mf.cent_to_hz([dist.bins[peak_idxs[min_row]]],
						             anti_freq)[0], dist_mat[min_row][min_col])
				elif(metric=='pd'):
					tonic_list[r] = (mf.cent_to_hz([shift_idxs[min_row] * self.cent_ss],
						             ref_freq)[0], dist_mat[min_row][min_col])
				# Current mode estimate is recorded.
				mode_list[r] = (mode_names[min_col], dist_mat[min_row][min_col])
				# The minimum value is replaced with a value larger than maximum,
				# so we won't return this estimate pair twice.
				dist_mat[min_row][min_col] = (np.amax(dist_mat) + 1)
			return mode_list, tonic_list

		# Tonic Estimation
		elif(est_tonic):
			# This part assigns the special case changes to standard variables,
			# so that we can treat PD and PCD in the same way, as much as
			# possible. 
			peak_idxs = shift_idxs if (metric=='pd') else peak_idxs
			ref_freq = anti_freq if (metric=='pcd') else ref_freq

			# Distance vector is generated. In the mode_estimate() function
			# of ModeFunctions, PD and PCD are treated differently and it
			# handles the special cases such as zero-padding. The mode is
			# already known, so there is only one model to be compared. Each
			# entry corresponds to one tonic candidate.
			distance_vector = mf.tonic_estimate(dist, peak_idxs, mode_dist,
				                                distance_method=distance_method,
				                                metric=metric, cent_ss=self.cent_ss)
			
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
				#TODO: review here, this might be tedious due to 257th line.
				if(metric=='pcd'):
					tonic_list[r] = (mf.cent_to_hz([dist.bins[peak_idxs[idx]]],
						             anti_freq)[0], distance_vector[idx])
				elif(metric=='pd'):
					tonic_list[r] = (mf.cent_to_hz([shift_idxs[idx] * self.cent_ss],
						             ref_freq)[0], distance_vector[idx])
				# Current minima is replaced with a value larger than maxima,
				# so that we won't return the same estimate twice.
				distance_vector[idx] = (np.amax(distance_vector) + 1)
			return tonic_list

		# Mode Estimation
		elif(est_mode):
			# Distance vector is generated. Again, mode_estimate() of
			# ModeFunctions handles the different approach required for
			# PCD and PD. Since tonic is known, the distributions aren't
			# shifted and are only compared to candidate mode models.
			distance_vector = mf.mode_estimate(dist, mode_dists, distance_method=distance_method, metric=metric, cent_ss=self.cent_ss)

			# Distance vector is ready now. For each rank, the loop is iterated.
			# When the first best estimate is found it's changed to be the worst,
			# so in the next iteration, the estimate would be the second best
			# and so on
			for r in range(min(rank, len(mode_names))):
				# Minima is found, corresponding mode candidate is our current
				# mode estimate
				idx = np.argmin(distance_vector)
				mode_list[r] = (mode_names[idx], distance_vector[idx])
				# Current minima is replaced with a value larger than maxima,
				# so that we won't return the same estimate twice.
				distance_vector[idx] = (np.amax(distance_vector) + 1)
			return mode_list
	
		else:
			# Nothing is expected to be estimated.
			return 0