# -*- coding: utf-8 -*-
import numpy as np
import ModeFunctions as mf
import PitchDistribution as p_d
import json
import os

class ChordiaEstimation:
	"""-------------------------------------------------------------------------
	This is an implementation of the method proposed for tonic and raag
	estimation, in the following paper. This also includes some extra features
	to the proposed version; such as the choice of using PD as well as PCD and
	the choice of fine-grained distributions as well as the smoothened ones. 

	* Chordia, P. and Şentürk, S. 2013. "Joint recognition of raag and tonic
	in North Indian music. Computer Music Journal", 37(3):82–98.

	We require a set of recordings with annotated modes and tonics to train the
	mode models. Unlike BozkurtEstimation, there is no single model for a mode.
	Instead, we slice pitch tracks into chunks and generate distributions for
	them. So, there are many sample points for each mode. 

	Then, the unknown mode and/or tonic of an input recording is estimated by
	comparing it to these models. For each chunk, we consider the close neighbors
	and based on the united set of close neighbors of all chunks of a recording,
	we apply K Nearest Neighbors to give a final decision about the whole
	recording. 
	-------------------------------------------------------------------------"""

	def __init__(self, cent_ss=7.5, smooth_factor=7.5, chunk_size=60,
		         threshold=0.5, overlap=0, hop_size=0.0029025):
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
		threshold     : This is the ratio of smallest acceptable chunk to chunk_size.
		                When a pitch track is sliced the remaining tail at its end is
		                returned if its longer than threshold*chunk_size. Else, it's
		                discarded. However if the entire track is shorter than this
		                it is still returned as it is, in order to be able to
		                represent that recording. 
		overlap       : If it's zero, the next chunk starts from the end of the
		                previous chunk, else it starts from the
		                (chunk_size*threshold)th sample of the previous chunk.
		hop_size      : The step size of timestamps of pitch tracks. This is used
		                for both training and estimating.
		------------------------------------------------------------------------"""
		self.cent_ss = cent_ss
		self.overlap = overlap
		self.smooth_factor = smooth_factor
		self.chunk_size = chunk_size
		self.threshold = threshold
		self.hop_size = hop_size

	def train(self, mode_name, pt_list, ref_freq_list, metric='pcd',
		      save_dir='./', pt_dir='./'):
		"""-------------------------------------------------------------------------
		For the mode trainings, the requirements are a set of recordings with 
		annotated tonics for each mode under consideration. This function only
		expects the recordings' pitch tracks and corresponding tonics as lists.
		The two lists should be indexed in parallel, so the tonic of ith pitch
		track in the pitch track list should be the ith element of tonic list.

		Each pitch track would be sliced into chunks of size chunk_size and their
		pitch distributions are generated. Then, each of such chunk distributions
		are appended to a list. This list represents the mode by sample points as
		much as the number of chunks. So, the result is a list of PitchDistribution
		objects, i.e. list of structured dictionaries and this is what is saved. 
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
		save_name = mode_name + '.json'
		dist_list = []

		# Each pitch track is iterated over and its pitch distribution is generated
		# individually, then appended to dist_list. Notice that although we treat
		# each chunk individually, we use a single tonic annotation for each recording
		# so we assume that the tonic doesn't change throughout a recording.
		for pt in range(len(pt_list)):
			# Pitch track is loaded from local directory
			pitch_track = mf.load_track(pt_list[pt], pt_dir)
			time_track = np.arange(0, (self.hop_size*len(pitch_track)), self.hop_size)
			# Current pitch track is sliced into chunks.
			pts, chunk_data = mf.slice(time_track, pitch_track, pt_list[pt],
				                 self.chunk_size, self.threshold, self.overlap)
			# Each chunk is converted to cents
			pts = [mf.hz_to_cent(k, ref_freq=ref_freq_list[pt]) for k in pts]
			# This is a wrapper function. It iteratively generates the distribution
			# for each chunk and return it as a list. After this point, we only
			# need to save it. God bless modular programming!
			temp_list = self.train_chunks(pts, chunk_data, ref_freq_list[pt], metric)
			# The list is composed of lists of PitchDistributions. So,
			# each list in temp_list corresponds to a recording and each
			# PitchDistribution in that list belongs to a chunk. Since these
			# objects remember everything, we just flatten the list and make
			# life much easier. From now on, each chunk is treated as an individual
			# distribution, regardless of which recording it belongs to.
			for tmp in temp_list:
				dist_list.append(tmp)

		# Dump the list of dictionaries in a JSON file.
		dist_json = [{'bins':d.bins.tolist(), 'vals':d.vals.tolist(),
		              'kernel_width':d.kernel_width, 'source':d.source,
		              'ref_freq':d.ref_freq, 'segmentation':d.segmentation,
		              'overlap':d.overlap} for d in dist_list]
		with open(os.path.join(save_dir, save_name), 'w') as f:
			json.dump(dist_json, f, indent=2)
			f.close()

	def estimate(self, pitch_track, mode_names=[], mode_name='',
		         mode_dir='./', est_tonic=True, est_mode=True,
		         distance_method="euclidean", metric='pcd', ref_freq=440,
		         k_param=1):
		"""-------------------------------------------------------------------------
		In the estimation phase, the input pitch track is sliced into chunk and each
		chunk is compared with each candidate mode's each sample model, i.e. with 
		the distributions of each training recording's each chunk. This function is
		a wrapper, that handles decision making portion and the overall flow of the
		estimation process. Internally, segment estimate is called for generation of
		distance matrices and detecting neighbor distributions.

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
		k_param         : The k parameter of K Nearest Neighbors. 
		distance_method : The choice of distance methods. See distance() in
		                  ModeFunctions for more information.
		metric          : Whether the model should be octave wrapped (Pitch Class
		                  Distribution: PCD) or not (Pitch Distribution: PD)
		ref_freq        : Annotated tonic of the recording. If it's unknown, we use
		                  an arbitrary value, so this can be ignored.
		-------------------------------------------------------------------------"""
		# Pitch track is sliced into chunks.
		time_track = np.arange(0, (self.hop_size*len(pitch_track)), self.hop_size)
		pts, chunk_data = mf.slice(time_track, pitch_track, 'input', self.chunk_size,
			                 self.threshold, self.overlap)

		# Here's a neat trick. In order to return an estimation about the entire
		# recording based on our observations on individual chunks, we look at the
		# nearest neighbors of  union of all chunks. We are returning min_cnt
		# many number of closest neighbors from each chunk. To make sure that we
		# capture all of the nearest neighbors, we return a little more than
		# required and then treat the union of these nearest neighbors as if it's
		# the distance matrix of the entire recording.Then, we find the nearest
		# neighbors from the union of these from each chunk. This is quite an
		# overshoot, we only need min_cnt >= k_param. 

		### TODO: shrink this value as much as possible.
		min_cnt = len(pts) * k_param

		#Initializations
		tonic_list = 0
		mode_list = ''

		if(est_tonic and est_mode):
			neighbors = [ [mode_list, tonic_list] for i in range(len(chunk_data)) ]
		elif(est_tonic):
			neighbors = [ tonic_list for i in range(len(chunk_data)) ]
		elif(est_mode):
			neighbors = [ mode_list for i in range(len(chunk_data)) ]

		# chunk_estimate() generates the distributions of each chunk iteratively,
		# then compares it with all candidates and returns min_cnt closest neighbors
		# of each chunk to neighbors list.
		for p in range(len(pts)):
			neighbors[p] = self.chunk_estimate(pts[p], mode_names=mode_names,
				                                 mode_name=mode_name, mode_dir=mode_dir,
				                                 est_tonic=est_tonic, est_mode=est_mode,
				                                 distance_method=distance_method,
				                                 metric=metric, ref_freq=ref_freq,
				                                 min_cnt=min_cnt)
		
		### TODO: Clean up the spaghetti decision making part. The procedures
		### are quite repetitive. Wrap them up with a separate function.

		# Temporary variables used during the desicion making part.
		candidate_distances, candidate_ests, candidate_sources, kn_distances, kn_ests,
		kn_sources, idx_counts, elem_counts, res_distances, res_sources = ([] for i in range(10))

		# Joint estimation decision making. 
		if(est_mode and est_tonic):
			# Flattens the returned candidates and related data about them and
			# stores them into candidate_* variables. candidate_distances stores
			# the distance values, candidate_ests stores the mode/tonic pairs
			# candidate_sources stores the sources of the nearest neighbors.
			for i in xrange(len(pts)):
				for j in neighbors[i][1]:
					candidate_distances.append(j)
				for l in xrange(len(neighbors[i][0][1])):
					candidate_ests.append((neighbors[i][0][1][l], neighbors[i][0][0][l][0]))
					candidate_sources.append(neighbors[i][0][0][l][1])

			# Finds the nearest neighbors and fills all related data about
			# them to kn_* variables. Each of these variables have length k.
			# kn_distances stores the distance values, kn_ests stores
			# mode/tonic pairs, kn_sources store the name/id of the distribution
			# that gave rise to the corresponding distances.
			for k in xrange(k_param):
				idx = np.argmin(candidate_distances)
				kn_distances.append(candidate_distances[idx])
				kn_ests.append(candidate_ests[idx])
				kn_sources.append(candidate_sources[idx])
				candidate_distances[idx] = (np.amax(candidate_distances) + 1)
			
			# Counts the occurences of each candidate mode/tonic pair in
			# the K nearest neighbors. The result is our estimation. 
			for c in set(kn_ests):
				idx_counts.append(kn_ests.count(c))
				elem_counts.append(c)
			joint_estimation = elem_counts[np.argmax(idx_counts)]

			# We have concluded our estimation. Here, we retrieve the 
			# relevant data to this estimation; the sources and coresponding
			# distances.
			for m in xrange(len(kn_ests)):
				if (kn_ests[m] == joint_estimation):
					res_sources.append(kn_sources[m])
					res_distances.append(kn_distances[m])
			result = [joint_estimation, res_sources, res_distances]

		# Mode estimation decision making
		elif(est_mode):
			# Flattens the returned candidates and related data about them and
			# stores them into candidate_* variables. candidate_distances stores
			# the distance values, candidate_ests stores the candidate modes
			# candidate_sources stores the sources of the nearest neighbors.
			for i in xrange(len(pts)):
				for j in neighbors[i][1]:
					candidate_distances.append(j)
				for l in xrange(len(neighbors[i][0])):
					candidate_ests.append(neighbors[i][0][l][0])
					candidate_sources.append(neighbors[i][0][l][1])

			# Finds the nearest neighbors and fills all related data about
			# them to kn_* variables. Each of these variables have length k.
			# kn_distances stores the distance values, kn_ests stores
			# mode names, kn_sources store the name/id of the distributions
			# that gave rise to the corresponding distances.
			for k in xrange(k_param):
				idx = np.argmin(candidate_distances)
				kn_distances.append(candidate_distances[idx])
				kn_ests.append(candidate_ests[idx])
				kn_sources.append(candidate_sources[idx])
				candidate_distances[idx] = (np.amax(candidate_distances) + 1)

			# Counts the occurences of each candidate mode name in
			# the K nearest neighbors. The result is our estimation. 
			for c in set(kn_ests):
				idx_counts.append(kn_ests.count(c))
				elem_counts.append(c)
			mode_estimation = elem_counts[np.argmax(idx_counts)]

			# We have concluded our estimation. Here, we retrieve the 
			# relevant data to this estimation; the sources and coresponding
			# distances.
			for m in xrange(len(kn_ests)):
				if (kn_ests[m] == mode_estimation):
					res_sources.append(kn_sources[m])
					res_distances.append(kn_distances[m])
			result = [mode_estimation, res_sources, res_distances]


		# Tonic estimation decision making
		elif(est_tonic):
			# Flattens the returned candidates and related data about them and
			# stores them into candidate_* variables. candidate_distances stores
			# the distance values, candidate_ests stores the candidate peak 
			# frequencies, candidate_sources stores the sources of the nearest
			# neighbors.
			for i in xrange(len(pts)):
				for j in neighbors[i][1]:
					candidate_distances.append(j)
				for l in xrange(len(neighbors[i][0])):
					candidate_ests.append(neighbors[i][0][l][0])
					candidate_sources.append(neighbors[i][0][l][1])

			# Finds the nearest neighbors and fills all related data about
			# them to kn_* variables. Each of these variables have length k.
			# kn_distances stores the distance values, kn_ests stores
			# peak frequencies, kn_sources store the name/id of the
			# distributions that gave rise to the corresponding distances.
			for k in xrange(k_param):
				idx = np.argmin(candidate_distances)
				kn_distances.append(candidate_distances[idx])
				kn_ests.append(candidate_ests[idx])
				kn_sources.append(candidate_sources[idx])
				candidate_distances[idx] = (np.amax(candidate_distances) + 1)

			# Counts the occurences of each candidate tonic frequency in
			# the K nearest neighbors. The result is our estimation. 
			for c in set(kn_ests):
				idx_counts.append(kn_ests.count(c))
				elem_counts.append(c)
			tonic_estimation = elem_counts[np.argmax(idx_counts)]

			# We have concluded our estimation. Here, we retrieve the 
			# relevant data to this estimation; the sources and coresponding
			# distances.
			for m in xrange(len(kn_ests)):
				if (kn_ests[m] == tonic_estimation):
					res_sources.append(kn_sources[m])
					res_distances.append(kn_distances[m])
			result = [tonic_estimation, res_sources, res_distances]

		return result

	def chunk_estimate(self, pitch_track, mode_names=[], mode_name='', mode_dir='./',
		                 est_tonic=True, est_mode=True, distance_method="euclidean",
		                 metric='pcd', ref_freq=440, min_cnt=3):
		"""-------------------------------------------------------------------------
		This function is called by the wrapper estimate() function only. It gets a 
		pitch track chunk, generates its pitch distribution and compares it with the
		chunk distributions of the candidate modes. Then, finds the min_cnt nearest
		neighbors and returns them to estimate(), where these are used to make an
		estimation about the overall recording.
		----------------------------------------------------------------------------
		pitch_track     : Pitch track chunk of the input recording whose tonic and/or
		                  mode is to be estimated. This is only a 1-D list of frequency
		                  values.
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
		distance_method : The choice of distance methods. See distance() in
		                  ModeFunctions for more information.
		metric          : Whether the model should be octave wrapped (Pitch Class
		                  Distribution: PCD) or not (Pitch Distribution: PD)
		ref_freq        : Annotated tonic of the recording. If it's unknown, we use
		                  an arbitrary value, so this can be ignored.
		min_cnt         : The number of nearest neighbors of the current chunk to be
		                  returned. The details of this parameter and its implications
		                  are explained in the first lines of estimate().
		-------------------------------------------------------------------------"""
		# Preliminaries before the estimations
		# Cent-to-Hz covnersion is done and pitch distributions are generated
		cent_track = mf.hz_to_cent(pitch_track, ref_freq)
		dist = mf.generate_pd(cent_track, ref_freq=ref_freq,
			                  smooth_factor=self.smooth_factor, cent_ss=self.cent_ss)
		dist = mf.generate_pcd(dist) if (metric=='pcd') else dist
		# The model mode distribution(s) are loaded. If the mode is annotated and tonic
		# is to be estimated, only the model of annotated mode is retrieved.
		mode_collections = [self.load_collection(mode, dist_dir=mode_dir) for mode in mode_names]
		mode_dists = [d for col in mode_collections for d in col]
		mode_dist = self.load_collection(mode_name, dist_dir=mode_dir) if (mode_name!='') else None
		# cum_lens (cummulative lengths) keeps track of number of chunks retrieved from
		# each mode. So that we are able to find out which mode the best performed chunk
		# belongs to.
		cum_lens = np.cumsum([len(col) for col in mode_collections])
		#Initializations of possible output parameters
		tonic_list = [0 for x in range(min_cnt)]
		mode_list = ['' for x in range(min_cnt)]
		min_distance_list = np.zeros(min_cnt)

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
		### TODO: The first steps of joint estimation are very similar for both Bozkurt and
		### Chordia. We might squeeze them into a single function in ModeFunctions.
		if(est_tonic and est_mode):
			if(metric=='pcd'):
				# PCD doesn't require any prelimimary steps. Generates the distance matrix.
				# The rows are tonic candidates and columns are mode candidates.
				dist_mat = mf.generate_distance_matrix(dist, peak_idxs, mode_dists, method=distance_method)
			elif(metric=='pd'):
				# Since PD lengths aren't equal, zero padding is required and
				# tonic_estimate() of ModeFunctions just does that. It can handle only
				# a single column, so the columns of the matrix are iteratively generated
				dist_mat = np.zeros((len(shift_idxs), len(mode_dists)))
				for m in xrange(len(mode_dists)):
					dist_mat[:,m] = mf.tonic_estimate(dist, shift_idxs, mode_dists[m],
						                              distance_method=distance_method,
						                              metric=metric, cent_ss=self.cent_ss)

			# Distance matrix is ready now. Since we need to report min_cnt many
			# nearest neighbors, the loop is iterated min_cnt times and returns
			# one neighbor at each iteration, from closest to futher. When first
			# nearest neighbor is found it's changed to the worst, so in the
			# next iteration, the nearest would be the second nearest and so on.
			for r in xrange(min_cnt):
				# The minima of the distance matrix is found. This is to find
				# the current nearest neighbor chunk.
				min_row = np.where((dist_mat == np.amin(dist_mat)))[0][0]
				min_col = np.where((dist_mat == np.amin(dist_mat)))[1][0]
				# Due to the precaution step of PCD, the reference frequency is
				# changed. That's why it's treated differently than PD. Here,
				# the cent value of the tonic estimate is converted back to Hz.	
				if(metric=='pcd'):
					tonic_list[r] = mf.cent_to_hz([dist.bins[peak_idxs[min_row]]],
						                          anti_freq)[0]
				elif(metric=='pd'):
					tonic_list[r] = mf.cent_to_hz([shift_idxs[min_row] * self.cent_ss],
						                          ref_freq)[0]
				# We have found out which chunk is our nearest now. Here, we find out
				# which mode it belongs to, from cum_lens.
				mode_list[r] = (mode_names[min(np.where((cum_lens > min_col))[0])],
					           mode_dists[min_col].source[:-6])
				# To observe how close these neighbors are, we report their distances.
				# This doesn't affect the computation at all and it's just for the 
				# evaluating and understanding the behvaviour of the system. 
				min_distance_list[r] = dist_mat[min_row][min_col]
				# The minimum value is replaced with a value larger than maximum,
				# so we can easily find the second nearest neighbor.
				dist_mat[min_row][min_col] = (np.amax(dist_mat) + 1)
			return [[mode_list, tonic_list], min_distance_list.tolist()]

		# Tonic Estimation
		elif(est_tonic):
			# This part assigns the special case changes to standard variables,
			# so that we can treat PD and PCD in the same way, as much as
			# possible.
			peak_idxs = shift_idxs if metric=='pd' else peak_idxs
			anti_freq = ref_freq if metric=='pd' else anti_freq

			# Distance matrix is generated. In the mode_estimate() function
			# of ModeFunctions, PD and PCD are treated differently and it
			# handles the special cases such as zero-padding. The mode is
			# already known, so there is only one mode collection, i.e.
			# set of chunk distributions that belong to the same mode, to
			# be compared. Each column is a chunk distribution and each
			# row is a tonic candidate.
			dist_mat = [mf.tonic_estimate(dist, peak_idxs, d,
				                          distance_method=distance_method,
				                          metric=metric, cent_ss=self.cent_ss) for d in mode_dist]

			# Distance matrix is ready now. Since we need to report min_cnt many
			# nearest neighbors, the loop is iterated min_cnt times and returns
			# one neighbor at each iteration, from closest to futher. When first
			# nearest neighbor is found it's changed to the worst, so in the
			# next iteration, the nearest would be the second nearest and so on.
			for r in xrange(min_cnt):
				# The minima of the distance matrix is found. This is to find
				# the current nearest neighbor chunk.
				min_row = np.where((dist_mat == np.amin(dist_mat)))[0][0]
				min_col = np.where((dist_mat == np.amin(dist_mat)))[1][0]
				# The corresponding tonic candidate is found, based on the
				# current nearest neighbor and it's distance is recorded
				tonic_list[r] = (mf.cent_to_hz([dist.bins[peak_idxs[min_col]]],
					                           anti_freq)[0], mode_dists[min_row].source[:-6])
				min_distance_list[r] = dist_mat[min_row][min_col]
				# The minimum value is replaced with a value larger than maximum,
				# so we can easily find the second nearest neighbor.
				dist_mat[min_row][min_col] = (np.amax(dist_mat) + 1)
			return [tonic_list, min_distance_list.tolist()]

		# Mode estimation
		elif(est_mode):
			# Only in mode estimation, the distance matrix is actually a vector.
			# Since the tonic is annotated, the distribution isn't shifted and
			# compared to each chunk distribution of each candidate mode.
			# Again, mode_estimate() of ModeFunctions handles the different
			# approach required for PCD and PD.
			distance_vector = mf.mode_estimate(dist, mode_dists,
				                               distance_method=distance_method,
				                               metric=metric, cent_ss=self.cent_ss)
			
			# Distance matrix is ready now. Since we need to report min_cnt many
			# nearest neighbors, the loop is iterated min_cnt times and returns
			# one neighbor at each iteration, from closest to futher. When first
			# nearest neighbor is found it's changed to the worst, so in the
			# next iteration, the nearest would be the second nearest and so on.
			for r in xrange(min_cnt):
				# The minima of the distance matrix is found. This is to find
				# the current nearest neighbor chunk.
				idx = np.argmin(distance_vector)
				# We have found out which chunk is our nearest now. Here, we find out
				# which mode it belongs to, from cum_lens.
				mode_list[r] = (mode_names[min(np.where((cum_lens > idx))[0])],
					                                    mode_dists[idx].source[:-6])
				# The distance of the current nearest neighbors recorded. The details
				# of this step is explained in the end of the analogous loop in joint
				# estimation of thşs function.
				min_distance_list[r] = distance_vector[idx]
				# The minimum value is replaced with a value larger than maximum,
				# so we can easily find the second nearest neighbor.
				distance_vector[idx] = (np.amax(distance_vector) + 1) 
			return [mode_list, min_distance_list.tolist()]

		else:
			return 0

	def train_chunks(self, pts, chunk_data, ref_freq, metric='pcd'):
		"""-------------------------------------------------------------------------
		Gets the pitch track chunks of a recording, generates its pitch distribution
		and returns the PitchDistribution objects as a list. This function is called
		for each of the recordings in the training. The outputs of this function are
		combined in train() and the resultant mode model is obtained.
		----------------------------------------------------------------------------
		pts        : List of pitch tracks of chunks that belong to the same mode.
		             The pitch distributions of these are iteratively generated to
		             use as the sample points of the mode model
		chunk_data : The relevant data about the chunks; source, initial timestamp
		             and final timestamp. The format is the same as slice() of
		             ModeFunctions.
		ref_freq   : Reference frequency to be used in PD/PCD generation. Since this
		             the training function, this should be the annotated tonic of the
		             recording
		metric     : The choice of PCD or PD
		-------------------------------------------------------------------------"""
		dist_list = []
		# Iterates over the pitch tracks of a recording
		for idx in range(len(pts)):
			# Retrieves the relevant information about the current chunk
			src = chunk_data[idx][0]
			interval = (chunk_data[idx][1], chunk_data[idx][2])
			# PitchDistribution of the current chunk is generated
			dist = mf.generate_pd(pts[idx], ref_freq=ref_freq,
				                  smooth_factor=self.smooth_factor,
				                  cent_ss=self.cent_ss, source=src,
				                  segment=interval, overlap=self.overlap)
			if(metric=='pcd'):
				dist = mf.generate_pcd(dist)
			# The resultant pitch distributions are filled in the list to be returned
			dist_list.append(dist)
		return dist_list

	def load_collection(self, mode_name, dist_dir='./'):
		"""-------------------------------------------------------------------------
		Since each mode model consists of a list of PitchDistribution objects, the
		load() function from that class can't be used directly. This function loads
		JSON files that contain a list of PitchDistribution objects. This is used
		for retrieving the mode models in the beginning of estimation process.
		----------------------------------------------------------------------------
		mode_name : Name of the mode to be loaded. The name of the JSON file is
		            expected to be "mode_name.json" 
		dist_dir  : Directory where the JSON file is stored.
		-------------------------------------------------------------------------"""
		obj_list = []
		fname = mode_name + '.json'
		with open(os.path.join(dist_dir, fname)) as f:
			dist_list = json.load(f)

		# List of dictionaries is is iterated over to initialize a list of
		# PitchDistribution objects.
		for d in dist_list:
			obj_list.append(p_d.PitchDistribution(np.array(d['bins']),
				            np.array(d['vals']), kernel_width=d['kernel_width'],
				            source=d['source'], ref_freq=d['ref_freq'],
				            segment=d['segmentation'], overlap=d['overlap']))
		return obj_list