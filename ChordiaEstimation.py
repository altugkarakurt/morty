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
		         threshold=0.5, overlap=0):
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
		------------------------------------------------------------------------"""
		self.cent_ss = cent_ss
		self.overlap = overlap
		self.smooth_factor = smooth_factor
		self.chunk_size = chunk_size
		self.threshold = threshold

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
			cur = mf.load_track(pt_list[pt], pt_dir)
			time_track = cur[:,0]
			pitch_track = cur[:,1]
			# Current pitch track is sliced into chunks.
			pts, segs = mf.slice(time_track, pitch_track, pt_list[pt],
				                 self.chunk_size, self.threshold, self.overlap)
			# Each chunk is converted to cents
			pts = [mf.hz_to_cent(k, ref_freq=ref_freq_list[pt]) for k in pts]
			# This is a wrapper function. It iteratively generates the distribution
			# for each chunk and return it as a list. After this point, we only
			# need to save it. God bless modular programming!
			temp_list = self.train_segments(pts, segs, ref_freq_list[pt],
				                            save_dir, save_name, metric)
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

	def estimate(self, pitch_track, time_track, mode_names=[], mode_name='',
		         mode_dir='./', est_tonic=True, est_mode=True,
		         distance_method="euclidean", metric='pcd', ref_freq=440,
		         k_param=1):
		pts, segs = mf.slice(time_track, pitch_track, 'input', self.chunk_size,
			                 self.threshold, self.overlap)
		tonic_list = 0
		mode_list = ''
		min_cnt = len(pts) * k_param
		if(est_tonic and est_mode):
			neighbors = [ [mode_list, tonic_list] for i in range(len(segs)) ]
		elif(est_tonic):
			neighbors = [ tonic_list for i in range(len(segs)) ]
		elif(est_mode):
			neighbors = [ mode_list for i in range(len(segs)) ]

		for p in range(len(pts)):
			neighbors[p] = self.segment_estimate(pts[p], mode_names=mode_names,
				                                 mode_name=mode_name, mode_dir=mode_dir,
				                                 est_tonic=est_tonic, est_mode=est_mode,
				                                 distance_method=distance_method,
				                                 metric=metric, ref_freq=ref_freq,
				                                 min_cnt=min_cnt)
		
		candidate_distances, candidate_ests, candidate_sources, kn_distances, kn_ests,
		kn_sources, idx_counts, elem_counts, res_distances, res_sources = ([] for i in range(10))

		if(est_mode and est_tonic):
			for i in xrange(len(pts)):
				for j in neighbors[i][1]:
					candidate_distances.append(j)
				for l in xrange(len(neighbors[i][0][1])):
					candidate_ests.append((neighbors[i][0][1][l], neighbors[i][0][0][l][0]))
					candidate_sources.append(neighbors[i][0][0][l][1])

			for k in xrange(k_param):
				idx = np.argmin(candidate_distances)
				kn_distances.append(candidate_distances[idx])
				kn_ests.append(candidate_ests[idx])
				kn_sources.append(candidate_sources[idx])
				candidate_distances[idx] = (np.amax(candidate_distances) + 1)
			
			for c in set(kn_ests):
				idx_counts.append(kn_ests.count(c))
				elem_counts.append(c)
			joint_estimation = elem_counts[np.argmax(idx_counts)]

			for m in xrange(len(kn_ests)):
				if (kn_ests[m] == joint_estimation):
					res_sources.append(kn_sources[m])
					res_distances.append(kn_distances[m])
			result = [joint_estimation, res_sources, res_distances]

		elif(est_mode):
			for i in xrange(len(pts)):
				for j in neighbors[i][1]:
					candidate_distances.append(j)
				for l in xrange(len(neighbors[i][0])):
					candidate_ests.append(neighbors[i][0][l][0])
					candidate_sources.append(neighbors[i][0][l][1])

			for k in xrange(k_param):
				idx = np.argmin(candidate_distances)
				kn_distances.append(candidate_distances[idx])
				kn_ests.append(candidate_ests[idx])
				kn_sources.append(candidate_sources[idx])
				candidate_distances[idx] = (np.amax(candidate_distances) + 1)
			
			for c in set(kn_ests):
				idx_counts.append(kn_ests.count(c))
				elem_counts.append(c)
			mode_estimation = elem_counts[np.argmax(idx_counts)]

			for m in xrange(len(kn_ests)):
				if (kn_ests[m] == mode_estimation):
					res_sources.append(kn_sources[m])
					res_distances.append(kn_distances[m])
			result = [mode_estimation, res_sources, res_distances]

		elif(est_tonic):

			for i in xrange(len(pts)):
				for j in neighbors[i][1]:
					candidate_distances.append(j)
				for l in xrange(len(neighbors[i][0])):
					candidate_ests.append(neighbors[i][0][l][0])
					candidate_sources.append(neighbors[i][0][l][1])

			for k in xrange(k_param):
				idx = np.argmin(candidate_distances)
				kn_distances.append(candidate_distances[idx])
				kn_ests.append(candidate_ests[idx])
				kn_sources.append(candidate_sources[idx])
				candidate_distances[idx] = (np.amax(candidate_distances) + 1)

			for c in set(kn_ests):
				idx_counts.append(kn_ests.count(c))
				elem_counts.append(c)
			tonic_estimation = elem_counts[np.argmax(idx_counts)]

			for m in xrange(len(kn_ests)):
				if (kn_ests[m] == tonic_estimation):
					res_sources.append(kn_sources[m])
					res_distances.append(kn_distances[m])
			result = [tonic_estimation, res_sources, res_distances]

		return result

	def segment_estimate(self, pitch_track, mode_names=[], mode_name='', mode_dir='./',
		                 est_tonic=True, est_mode=True, distance_method="euclidean",
		                 metric='pcd', ref_freq=440, min_cnt=3):
		### Preliminaries before the estimations
		cent_track = mf.hz_to_cent(pitch_track, ref_freq)
		dist = mf.generate_pd(cent_track, ref_freq=ref_freq,
			                  smooth_factor=self.smooth_factor, cent_ss=self.cent_ss)
		dist = mf.generate_pcd(dist) if (metric=='pcd') else dist
		mode_collections = [self.load_collection(mode, metric, dist_dir=mode_dir) for mode in mode_names]
		cum_lens = np.cumsum([len(col) for col in mode_collections])
		mode_dists = [d for col in mode_collections for d in col]
		mode_dist = self.load_collection(mode_name, metric, dist_dir=mode_dir) if (mode_name!='') else None
		tonic_list = [0 for x in range(min_cnt)]
		mode_list = ['' for x in range(min_cnt)]
		min_distance_list = np.zeros(min_cnt)

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
				for m in xrange(len(mode_dists)):
					dist_mat[:,m] = mf.tonic_estimate(dist, shift_idxs, mode_dists[m],
						                              distance_method=distance_method,
						                              metric=metric, cent_ss=self.cent_ss)

			for r in xrange(min_cnt):
				min_row = np.where((dist_mat == np.amin(dist_mat)))[0][0]
				min_col = np.where((dist_mat == np.amin(dist_mat)))[1][0]	
				if(metric=='pcd'):
					tonic_list[r] = mf.cent_to_hz([dist.bins[peak_idxs[min_row]]],
						                          anti_freq)[0]
				elif(metric=='pd'):
					tonic_list[r] = mf.cent_to_hz([shift_idxs[min_row] * self.cent_ss],
						                          ref_freq)[0]
				mode_list[r] = (mode_names[min(np.where((cum_lens > min_col))[0])],
					           mode_dists[min_col].source[:-6])
				min_distance_list[r] = dist_mat[min_row][min_col]
				dist_mat[min_row][min_col] = (np.amax(dist_mat) + 1)
			return [[mode_list, tonic_list], min_distance_list.tolist()]

		elif(est_tonic):
			peak_idxs = shift_idxs if metric=='pd' else peak_idxs
			dist_mat = [mf.tonic_estimate(dist, peak_idxs, d,
				                          distance_method=distance_method,
				                          metric=metric, cent_ss=self.cent_ss) for d in mode_dist]
			anti_freq = ref_freq if metric=='pd' else anti_freq

			for r in xrange(min_cnt):
				min_row = np.where((dist_mat == np.amin(dist_mat)))[0][0]
				min_col = np.where((dist_mat == np.amin(dist_mat)))[1][0]
				tonic_list[r] = (mf.cent_to_hz([dist.bins[peak_idxs[min_col]]],
					                           anti_freq)[0], mode_dists[min_row].source[:-6])
				min_distance_list[r] = dist_mat[min_row][min_col]
				dist_mat[min_row][min_col] = (np.amax(dist_mat) + 1)
			return [tonic_list, min_distance_list.tolist()]

		elif(est_mode):
			distance_vector = mf.mode_estimate(dist, mode_dists,
				                               distance_method=distance_method,
				                               metric=metric, cent_ss=self.cent_ss)
			for r in xrange(min_cnt):
				idx = np.argmin(distance_vector)
				mode_list[r] = (mode_names[min(np.where((cum_lens > idx))[0])],
					                                    mode_dists[idx].source[:-6])
				min_distance_list[r] = distance_vector[idx]
				distance_vector[idx] = (np.amax(distance_vector) + 1) 
			return [mode_list, min_distance_list.tolist()]

		else:
			return 0

	def train_segments(self, pts, seg_tuples, ref_freq, save_dir,
		               save_name, metric='pcd'):
		dist_list = []
		for idx in range(len(pts)):
			src = seg_tuples[idx][0]
			interval = (seg_tuples[idx][1], seg_tuples[idx][2])
			dist = mf.generate_pd(pts[idx], ref_freq=ref_freq,
				                  smooth_factor=self.smooth_factor,
				                  cent_ss=self.cent_ss, source=src,
				                  segment=interval, overlap=self.overlap)
			if(metric=='pcd'):
				dist = mf.generate_pcd(dist)
			dist_list.append(dist)
		return dist_list

	def load_collection(self, mode_name, metric, dist_dir='./'):
		obj_list = []
		fname = mode_name + '.json'
		with open(os.path.join(dist_dir, fname)) as f:
			dist_list = json.load(f)
		for d in dist_list:
			obj_list.append(p_d.PitchDistribution(np.array(d['bins']),
				            np.array(d['vals']), kernel_width=d['kernel_width'],
				            source=d['source'], ref_freq=d['ref_freq'],
				            segment=d['segmentation'], overlap=d['overlap']))
		return obj_list