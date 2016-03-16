# -*- coding: utf-8 -*-
import numpy as np
from modetonicestimation import ModeFunctions as mf
from modetonicestimation.PitchDistribution import PitchDistribution
import json
import os
import random


class Chordia:
    """---------------------------------------------------------------------
    This is an implementation of the method proposed for tonic and raag
    estimation, in the following paper. This also includes some extra features
    to the proposed version; such as the choice of using PD as well as PCD and
    the choice of fine-grained distributions as well as the smoothened ones.


    * Chordia, P. and Şentürk, S. 2013. "Joint recognition of raag and tonic
    in North Indian music. Computer Music Journal", 37(3):82–98.

    We require a set of recordings with annotated modes and tonics to train the
    mode models. Unlike BozkurtEstimation, there is no single model for a mode.
    Instead, we slice_pitch_track pitch tracks into chunks and generate

    distributions for them. So, there are many sample points for each mode.


    Then, the unknown mode and/or tonic of an input recording is estimated by
    comparing it to these models. For each chunk, we consider the close

    neighbors and based on the united set of close neighbors of all chunks

    of a recording, we apply K Nearest Neighbors to give a final decision

    about the whole recording.

    ---------------------------------------------------------------------"""

    def __init__(self, step_size=7.5, smooth_factor=7.5, chunk_size=60,
                 threshold=0.5, overlap=0, frame_rate=128.0 / 44100):
        """--------------------------------------------------------------------
        These attributes are wrapped as an object since these are used in both

        training and estimation stages and must be consistent in both processes
        -----------------------------------------------------------------------
        step_size       : Step size of the distribution bins
        smooth_factor : Std. deviation of the gaussian kernel used to smoothen
                        the distributions. For further details,
                        see generate_pd() of ModeFunctions.
        chunk_size    : The size of the recording to be considered. If zero,
                        the entire recording will be used to generate the pitch
                        distributions. If this is t, then only the first t
                        seconds of the recording is used only and remaining
                        is discarded.
        threshold     : This is the ratio of smallest acceptable chunk to
                        chunk_size. When a pitch track is sliced the
                        remaining tail at its end is returned if its longer
                        than threshold*chunk_size. Else, it's discarded.
                        However if the entire track is shorter than this
                        it is still returned as it is, in order to be able to
                        represent that recording.

        overlap       : If it's zero, the next chunk starts from the end of the
                        previous chunk, else it starts from the
                        (chunk_size*threshold)th sample of the previous chunk.
        frame_rate      : The step size of timestamps of pitch tracks. Default
                        is (128 = hopSize of the pitch extractor in
                        predominantmelodymakam repository) divided by 44100
                        audio sampling frequency. This is used to
                        slice_pitch_track the pitch tracks according to the
                        given chunk_size.
        --------------------------------------------------------------------"""
        self.step_size = step_size
        self.overlap = overlap
        self.smooth_factor = smooth_factor
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.frame_rate = frame_rate

    def train(self, mode_name, pitch_files, tonic_freqs, metric='pcd',
              save_dir=''):
        """--------------------------------------------------------------------
        For the mode trainings, the requirements are a set of recordings with

        annotated tonics for each mode under consideration. This function only
        expects the recordings' pitch tracks and corresponding tonics as lists.
        The two lists should be indexed in parallel, so the tonic of ith pitch
        track in the pitch track list should be the ith element of tonic list.

        Each pitch track would be sliced into chunks of size chunk_size and
        their pitch distributions are generated. Then, each of such chunk
        distributions are appended to a list. This list represents the mode
        by sample points as much as the number of chunks. So, the result is
        a list of PitchDistribution objects, i.e. list of structured
        dictionaries and this is what is saved.
        -----------------------------------------------------------------------
        mode_name     : Name of the mode to be trained. This is only used for
                        naming the resultant JSON file, in the form
                        "mode_name.json"
        pitch_files   : List of pitch tracks (i.e. 1-D list of frequencies)
        tonic_freqs   : List of annotated tonics of recordings
        metric        : Whether the model should be octave wrapped (Pitch Class
                        Distribution: PCD) or not (Pitch Distribution: PD)
        save_dir      : Where to save the resultant JSON files.
        --------------------------------------------------------------------"""
        pitch_distrib_list = []

        # Each pitch track is iterated over and its pitch distribution is
        # generated individually, then appended to pitch_distrib_list.
        # Notice that although we treat each chunk individually, we use a
        # single tonic annotation for each recording so we assume that the
        #  tonic doesn't change throughout a recording.
        tonic_freqs = [np.array(tonic) for tonic in tonic_freqs]
        pitch_tracks = mf.parse_pitch_track(pitch_files, multiple=True)
        for pitch_track, tonic in zip(pitch_tracks, tonic_freqs):
            time_track = np.arange(0, (self.frame_rate * len(pitch_track)),
                                   self.frame_rate)

            # Current pitch track is sliced into chunks.
            if not self.chunk_size:  # no slicing
                chunks = [pitch_track]
            else:
                chunks = mf.slice_pitch_track(
                    time_track, pitch_track, self.chunk_size,
                    self.threshold, self.overlap)

            # Each chunk is converted to cents
            chunks = [mf.hz_to_cent(k, ref_freq=tonic) for k in chunks]

            # This is a wrapper function. It iteratively generates the
            # distribution for each chunk and return it as a list. After
            # this point, we only need to save it.
            temp_list = self.train_chunks(chunks, tonic, metric)

            # The list is composed of lists of PitchDistributions. So,
            # each list in temp_list corresponds to a recording and each
            # PitchDistribution in that list belongs to a chunk. Since these
            # objects remember everything, we just flatten the list and make
            # life much easier. From now on, each chunk is treated as an
            # individual distribution, regardless of which recording it
            # belongs to.
            for tmp in temp_list:
                pitch_distrib_list.append(tmp)

        # save the model to a file, if requested
        if save_dir:
            Chordia.save_model(pitch_distrib_list, save_dir, mode_name)

        return pitch_distrib_list

    @classmethod
    def load_model(cls, mode_name, dist_dir='./'):
        """--------------------------------------------------------------------
        Since each mode model consists of a list of PitchDistribution objects,
        the load() function from that class can't be used directly. This
        function loads JSON files that contain a list of PitchDistribution
        objects. This is used for retrieving the mode models in the
        beginning of estimation process.
        -----------------------------------------------------------------------
        mode_name : Name of the mode to be loaded. The name of the JSON file is
                    expected to be "mode_name.json"
        dist_dir  : Directory where the JSON file is stored.
        --------------------------------------------------------------------"""
        obj_list = []
        fname = mode_name + '.json'
        dist_list = json.load(open(os.path.join(dist_dir, fname)))

        # List of dictionaries is is iterated over to initialize a list of
        # PitchDistribution objects.
        for d in dist_list:
            obj_list.append(PitchDistribution(
                np.array(d['bins']), np.array(d['vals']),
                kernel_width=d['kernel_width'], ref_freq=d['ref_freq']))
        return obj_list

    @classmethod
    def save_model(cls, distribution_list, save_dir, mode_name):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Dump the list of dictionaries in a JSON file.
        dist_json = [{'bins': d.bins.tolist(), 'vals': d.vals.tolist(),
                      'kernel_width': d.kernel_width,
                      'ref_freq': d.ref_freq.tolist()}
                     for d in distribution_list]

        json.dump(dist_json, open(os.path.join(save_dir, mode_name + '.json'),
                                  'w'), indent=2)

    def joint_estimate(self, pitch_file, mode_names='', mode_dir='./',
                       distance_method="bhat", metric='pcd', k_param=3,
                       equal_sample_per_mode=False):
        """--------------------------------------------------------------------
        In the estimation phase, the input pitch track is sliced into chunk and
        each chunk is compared with each candidate mode's each sample model,

        i.e. with the distributions of each training recording's each chunk.

        This function is a wrapper, that handles decision making portion and

        the overall flow of the estimation process. Internally, segment

        estimate is called for generation of distance matrices and detecting

        neighbor distributions.

        Joint Estimation: Neither the tonic nor the mode of the recording is

        known. Then, joint estimation estimates both of these parameters

        without any prior knowledge about the recording.
        To use this: est_mode and est_tonic flags should be True since both are
        to be estimated. In this case tonic_freq and mode_name parameters

        are not used, since these are used to pass the annotated data about

        the recording.
        -----------------------------------------------------------------------
        pitch_file      : File in which the pitch track of the input recording
                        whose tonic and/or mode is to be estimated.
        mode_dir        : The directory where the mode models are stored. This

                        is to load the annotated mode or the candidate mode.
        mode_names      : Names of the candidate modes. These are used when

                        loading the mode models. If the mode isn't

                        estimated, this parameter isn't used and can be

                        ignored.
        mode_name       : Annotated mode of the recording. If it's not known

                        and to be estimated, this parameter isn't used and

                        can be ignored.
        k_param         : The k parameter of K Nearest Neighbors.

        distance_method : The choice of distance methods. See distance() in
                        ModeFunctions for more information.
        metric          : Whether the model should be octave wrapped (Pitch

                        Class Distribution: PCD) or not (Pitch Distribution:

                        PD)
        tonic_freq      : Annotated tonic of the recording. If it's unknown,

                        we use an arbitrary value, so this can be ignored.
        --------------------------------------------------------------------"""

        # load pitch track

        pitch_track = mf.parse_pitch_track(pitch_file, multiple=False)

        # Pitch track is sliced into chunks.
        time_track = np.arange(0, (self.frame_rate * len(pitch_track)),
                               self.frame_rate)

        if not self.chunk_size:  # no slicing
            chunks = [pitch_track]
        else:
            chunks = mf.slice_pitch_track(
                time_track, pitch_track, self.chunk_size, self.threshold,
                self.overlap)

        # Here's a neat trick. In order to return an estimation about the
        # entire recording based on our observations on individual chunks,
        # we look at the nearest neighbors of  union of all chunks. We are
        # returning min_cnt many number of closest neighbors from each
        # chunk. To make sure that we capture all of the nearest neighbors,
        # we return a little more than required and then treat the union of
        # these nearest neighbors as if it's the distance matrix of the
        # entire recording.Then, we find the nearest neighbors from the
        # union of these from each chunk. This is quite an overshoot,
        # we only need min_cnt >= k_param.

        min_cnt = len(chunks) * k_param

        neighbors = [['', 0] for _ in chunks]

        # chunk_estimate() generates the distributions of each chunk
        # iteratively, then compares it with all candidates and returns
        # min_cnt closest neighbors of each chunk to neighbors list.
        for p, _ in enumerate(chunks):
            neighbors[p] = self.chunk_estimate(
                chunks[p], mode_names=mode_names, mode_dir=mode_dir,
                est_tonic=True, est_mode=True, distance_method=distance_method,
                metric=metric, min_cnt=min_cnt,
                equal_sample_per_mode=equal_sample_per_mode)

        # The decision making about the entire recording starts here. neighbors
        # parameter holds the nearest neighbors to each chunk. We need to
        # process them as if they are from a single point in the space.

        # Temporary variables used during the desicion making part.
        # candidate_foo : the flattened list of all foos from chunks
        # kn_foo : k nearest foos of candidate_foos. this is a subset of
        #           candidate_foo
        candidate_distances, candidate_ests, kn_distances, kn_ests, = (
            [] for _ in range(4))

        # Joint estimation decision making.
        # Flattens the returned candidates and related data about them and
        # stores them into candidate_*
        for i, _ in enumerate(chunks):
            candidate_distances.extend(neighbors[i][1])
            for l, _ in enumerate(neighbors[i][0][1]):
                candidate_ests.append(
                    (neighbors[i][0][1][l], neighbors[i][0][0][l]))

        # Finds the nearest neighbors and fills all related data about
        # them to kn_*. Each of these variables have length k. kn_distances
        # stores the distance values, kn_ests stores mode/tonic pairs.
        for k in range(k_param):
            idx = np.argmin(candidate_distances)
            kn_distances.append(candidate_distances[idx])
            kn_ests.append(candidate_ests[idx])
            candidate_distances[idx] = (np.amax(candidate_distances) + 1)

        # Counts the occurences of each candidate mode/tonic pair in
        # the K nearest neighbors. The result is our estimation.
        elem_counts = [c for c in set(kn_ests)]
        idx_counts = [kn_ests.count(c) for c in set(kn_ests)]
        return elem_counts[np.argmax(idx_counts)]

    def tonic_estimate(self, pitch_file, mode_name, mode_dir='./',
                       distance_method="bhat", metric='pcd', k_param=3,
                       equal_sample_per_mode=False):
        """--------------------------------------------------------------------
        Tonic Estimation: The mode of the recording is known and tonic is to be
        estimated. This is generally the most accurate estimation among the
        three. To use this: est_tonic should be True and est_mode should be
        False. In this case tonic_freq  and mode_names parameters are not
        used since tonic isn't known a priori and mode is known and hence
        there is no candidate mode.
        -----------------------------------------------------------------------
        See joint_estimate() for details. The I/O part of *_estimate()
        functions are identical.
        --------------------------------------------------------------------"""
        # load pitch track

        pitch_track = mf.parse_pitch_track(pitch_file, multiple=False)

        # Pitch track is sliced into chunks.
        time_track = np.arange(0, self.frame_rate * len(pitch_track),
                               self.frame_rate)

        if not self.chunk_size:  # no slicing
            chunks = [pitch_track]
        else:
            chunks = mf.slice_pitch_track(
                time_track, pitch_track, self.chunk_size, self.threshold,
                self.overlap)

        min_cnt = len(chunks) * k_param

        neighbors = [0 for _ in chunks]

        for p, _ in enumerate(chunks):
            neighbors[p] = self.chunk_estimate(
                chunks[p], mode_name=mode_name, mode_dir=mode_dir,
                est_tonic=True, est_mode=False,
                distance_method=distance_method, metric=metric,
                min_cnt=min_cnt, equal_sample_per_mode=equal_sample_per_mode)

        candidate_distances, candidate_ests, kn_distances, kn_ests = (
            [] for _ in range(4))

        # See the joint version of this loop for further explanation
        for i in range(len(chunks)):
            candidate_distances.extend(neighbors[i][1])
            candidate_ests.extend(neighbors[i][0])

        for k in range(k_param):
            idx = np.argmin(candidate_distances)
            kn_distances.append(candidate_distances[idx])
            kn_ests.append(candidate_ests[idx])
            candidate_distances[idx] = (np.amax(candidate_distances) + 1)

        elem_counts = [c for c in set(kn_ests)]
        idx_counts = [kn_ests.count(c) for c in set(kn_ests)]
        return elem_counts[np.argmax(idx_counts)]

    def mode_estimate(self, pitch_file, tonic_freq, mode_names, mode_dir='./',
                      distance_method="bhat", metric='pcd',
                      k_param=3, equal_sample_per_mode=False):
        """--------------------------------------------------------------------
        Mode Estimation: The tonic of the recording is known and mode is to be
        estimated.
        To use this: est_mode should be True and est_tonic should be False. In

        this case mode_name parameter isn't used since the mode annotation

        is not available. It can be ignored.
        -----------------------------------------------------------------------
        See joint_estimate() for details. The I/O part of *_estimate()

        functions are identical.
        --------------------------------------------------------------------"""
        # load pitch track

        pitch_track = mf.parse_pitch_track(pitch_file, multiple=False)

        # Pitch track is sliced into chunks.
        time_track = np.arange(0, (self.frame_rate * len(pitch_track)),
                               self.frame_rate)

        if not self.chunk_size:  # no slicing
            chunks = [pitch_track]
        else:
            chunks = mf.slice_pitch_track(
                time_track, pitch_track, self.chunk_size, self.threshold,
                self.overlap)

        min_cnt = len(chunks) * k_param

        neighbors = ['' for _ in chunks]

        for p, _ in enumerate(chunks):
            neighbors[p] = self.chunk_estimate(
                chunks[p], mode_names=mode_names, mode_dir=mode_dir,
                est_tonic=False, est_mode=True,
                distance_method=distance_method, metric=metric,
                ref_freq=tonic_freq, min_cnt=min_cnt,
                equal_sample_per_mode=equal_sample_per_mode)

        candidate_distances, candidate_ests, kn_distances, kn_ests, = (
            [] for _ in range(4))

        # See the joint version of this loop for further explanation
        for i, _ in enumerate(chunks):
            candidate_distances.extend(neighbors[i][1])
            candidate_ests.extend(neighbors[i][0])

        for k in range(k_param):
            idx = np.argmin(candidate_distances)
            kn_distances.append(candidate_distances[idx])
            kn_ests.append(candidate_ests[idx])
            candidate_distances[idx] = (np.amax(candidate_distances) + 1)

        elem_counts = [c for c in set(kn_ests)]
        idx_counts = [kn_ests.count(c) for c in set(kn_ests)]
        return elem_counts[np.argmax(idx_counts)]

    def chunk_estimate(self, pitch_track, mode_names=None, mode_name='',
                       mode_dir='./', est_tonic=True, est_mode=True,
                       distance_method="bhat", metric='pcd', ref_freq=440,
                       min_cnt=3, equal_sample_per_mode=False):
        """--------------------------------------------------------------------
        This function is called by the wrapper estimate() function only. It
        gets a pitch track chunk, generates its pitch distribution and compares
        it with the chunk distributions of the candidate modes. Then,
        finds the min_cnt nearest neighbors and returns them to estimate(),
        where these are used to make an estimation about the overall recording.
        -----------------------------------------------------------------------
        pitch_track     : Pitch track chunk of the input recording whose tonic
                        and/or mode is to be estimated. This is only a 1-D
                        list of frequency values.
        mode_dir        : The directory where the mode models are stored. This
                        is to load the annotated mode or the candidate mode.
        mode_names      : Names of the candidate modes. These are used when
                        loading the mode models. If the mode isn't
                        estimated, this parameter isn't used and can be
                        ignored.
        mode_name       : Annotated mode of the recording. If it's not known
                        and to be estimated, this parameter isn't used and
                        can be ignored.
        est_tonic       : Whether tonic is to be estimated or not. If this flag
                        is False, ref_freq is treated as the annotated tonic.
        est_mode        : Whether mode is to be estimated or not. If this flag
                        is False, mode_name is treated as the annotated mode.
        distance_method : The choice of distance methods. See distance() in
                          ModeFunctions for more information.
        metric          : Whether the model should be octave wrapped (Pitch
                        Class Distribution: PCD) or not (Pitch Distribution:
                        PD)
        ref_freq        : Annotated tonic of the recording. If it's unknown, we
                        use an arbitrary value, so this can be ignored.
        min_cnt         : The number of nearest neighbors of the current
                        chunk to be returned. The details of this parameter
                        and its implications are explained in the first
                        lines of estimate().
        --------------------------------------------------------------------"""
        # Preliminaries before the estimations
        # Cent-to-Hz covnersion is done and pitch distributions are generated
        cent_track = mf.hz_to_cent(pitch_track, ref_freq)
        dist = PitchDistribution.from_cent_pitch(
            cent_track, ref_freq=ref_freq, smooth_factor=self.smooth_factor,
            step_size=self.step_size)
        dist = dist.to_pcd() if metric == 'pcd' else dist
        # The model mode distribution(s) are loaded. If the mode is annotated
        # and tonic is to be estimated, only the model of annotated mode is
        # retrieved.
        mode_collections = [Chordia.load_model(mode, dist_dir=mode_dir)
                            for mode in mode_names]

        # This is used when we want to use equal number of points as model
        # per each candidate mode. This comes in handy if the numbers of

        # chunks in trained mode model aren't close to each other.
        if equal_sample_per_mode:
            min_samp = min([len(n) for n in mode_collections])
            for i, m in enumerate(mode_collections):
                mode_collections[i] = random.sample(m, min_samp)

        # cum_lens (cummulative lengths) keeps track of number of chunks
        # retrieved from each mode. So that we are able to find out which is
        #  the mode of the closest chunk.
        cum_lens = np.cumsum([len(col) for col in mode_collections])

        # load mode distribution
        mode_dists = [d for col in mode_collections for d in col]
        mode_dist = Chordia.load_model(mode_name, dist_dir=mode_dir) \
            if mode_name != '' else None

        # Initializations of possible output parameters
        tonic_list = [0 for x in range(min_cnt)]
        mode_list = ['' for x in range(min_cnt)]
        min_distance_list = np.zeros(min_cnt)

        # If tonic will be estimated, there are certain common preliminary
        # steps, regardless of the process being a joint estimation of a
        # tonic estimation.
        if est_tonic:
            if metric == 'pcd':
                # This is a precaution. If there happens to be a peak at the
                # last (and first due to the circular nature of PCD) sample,
                # it's considered as two peaks, one at the end and one at
                # the beginning. To prevent this, we find the global minima
                # of the distribution and shift it to the beginning,
                # i.e. make it the new reference frequency. This new
                # reference could have been any other as long as there is no
                # peak there, but minima is fairly easy to find.
                shift_factor = dist.vals.tolist().index(min(dist.vals))
                dist = dist.shift(shift_factor)

                # new_ref_freq is the new reference frequency after shift,
                # as mentioned above.
                new_ref_freq = mf.cent_to_hz([dist.bins[shift_factor]],
                                             ref_freq=ref_freq)[0]
                # Peaks of the distribution are found and recorded. These will
                # be treated as tonic candidates.
                peak_idxs, peak_vals = dist.detect_peaks()

            elif metric == 'pd':
                # Since PD isn't circular, the precaution in PCD is unnecessary
                # here. Peaks of the distribution are found and recorded.
                # These will be treated as tonic candidates.
                peak_idxs, peak_vals = dist.detect_peaks()
                # The number of samples to be shifted is the list
                # [peak indices - zero bin] origin is the bin with value
                # zero and the shifting is done w.r.t. it.
                origin = np.where(dist.bins == 0)[0][0]
                shift_idxs = [(idx - origin) for idx in peak_idxs]

        # Here the actual estimation steps begin

        # Joint Estimation
        # TODO: The first steps of joint estimation are very similar for both
        # Bozkurt and Chordia. We might squeeze them into a single function
        # in ModeFunctions.
        if est_tonic and est_mode:
            if metric == 'pcd':
                # PCD doesn't require any prelimimary steps. Generates the
                # distance matrix. The rows are tonic candidates and columns
                #  are mode candidates.
                dist_mat = mf.generate_distance_matrix(
                    dist, peak_idxs, mode_dists, method=distance_method)
            elif metric == 'pd':
                # Since PD lengths aren't equal, zero padding is required and
                # tonic_estimate() of ModeFunctions just does that. It can
                # handle only a single column, so the columns of the matrix
                # are iteratively generated
                dist_mat = np.zeros((len(shift_idxs), len(mode_dists)))
                for m, cur_mode_dist in enumerate(mode_dists):
                    dist_mat[:, m] = mf.tonic_estimate(
                        dist, shift_idxs, cur_mode_dist,
                        distance_method=distance_method, metric=metric,
                        step_size=self.step_size)

            # Since we need to report min_cnt many nearest neighbors, the loop

            # is iterated min_cnt times and returns a neighbor at each
            # iteration, from closest to futher. When first nearest neighbor
            #  is found it's changed to be the furthest, so in the next
            # iteration, the nearest would be the second nearest and so on.
            for r in range(min_cnt):
                # The minima of the distance matrix is found. This is to find
                # the current nearest neighbor chunk.
                min_row = np.where((dist_mat == np.amin(dist_mat)))[0][0]
                min_col = np.where((dist_mat == np.amin(dist_mat)))[1][0]
                # Due to the precaution step of PCD, the reference frequency is
                # changed. The cent value of the tonic estimate is converted
                # back to Hz.
                if metric == 'pcd':
                    tonic_list[r] = mf.cent_to_hz(
                        [dist.bins[peak_idxs[min_row]]], new_ref_freq)[0]
                elif metric == 'pd':
                    tonic_list[r] = mf.cent_to_hz(
                        [shift_idxs[min_row] * self.step_size], ref_freq)[0]
                # We have found out which chunk is our nearest now. Here, we
                # find out which mode it belongs to, using cum_lens. The -6
                # index is for cleaning '.pitch' extension of the pitch
                # track file's name
                mode_list[r] = mode_names[min(np.where(cum_lens > min_col)[0])]

                # We report their distances just for forensics. They don't
                # affect the estimation.
                min_distance_list[r] = dist_mat[min_row][min_col]
                # The minimum value is replaced to be the max as explained
                # above
                dist_mat[min_row][min_col] = (np.amax(dist_mat) + 1)
            return [[mode_list, tonic_list], min_distance_list.tolist()]

        # Tonic Estimation
        elif est_tonic:
            # This part assigns the special case changes to standard variables,
            # so PD and PCD can be treated in the same way.
            peak_idxs = shift_idxs if metric == 'pd' else peak_idxs
            new_ref_freq = ref_freq if metric == 'pd' else new_ref_freq

            # Distance matrix is generated. The mode is already known,
            # so there is only one mode collection, i.e. set of chunk

            # distributions belong to the same mode. Each column is for

            # a chunk distribution and each row is for a tonic candidate.
            dist_mat = [mf.tonic_estimate(
                dist, peak_idxs, d, distance_method=distance_method,
                metric=metric, step_size=self.step_size) for d in mode_dist]

            # See the joint estimation version of this loop for further
            # explanations
            for r in range(min_cnt):
                min_row = np.where((dist_mat == np.amin(dist_mat)))[0][0]
                min_col = np.where((dist_mat == np.amin(dist_mat)))[1][0]
                tonic_list[r] = mf.cent_to_hz([dist.bins[peak_idxs[min_col]]],
                                              new_ref_freq)[0]
                min_distance_list[r] = dist_mat[min_row][min_col]
                dist_mat[min_row][min_col] = (np.amax(dist_mat) + 1)
            return [tonic_list, min_distance_list.tolist()]

        # Mode estimation
        elif est_mode:
            # Only in mode estimation, the distance matrix is actually a vector
            # Since the tonic is annotated, the distribution isn't shifted and
            # compared to each chunk distribution in each candidate mode model.
            distance_vector = mf.mode_estimate(
                dist, mode_dists, distance_method=distance_method,
                metric=metric, step_size=self.step_size)

            # See the joint estimation version of this loop for further
            # explanations.
            for r in range(min_cnt):
                idx = np.argmin(distance_vector)
                mode_list[r] = mode_names[min(np.where((cum_lens > idx))[0])]
                min_distance_list[r] = distance_vector[idx]
                distance_vector[idx] = (np.amax(distance_vector) + 1)
            return [mode_list, min_distance_list.tolist()]

    def train_chunks(self, pts, ref_freq, metric='pcd'):
        """--------------------------------------------------------------------
        Gets the pitch track chunks of a recording, generates its pitch
        distribution and returns the PitchDistribution objects as a list.
        This function is called for each of the recordings in the training.
        The outputs of this function are combined in train() and the
        resultant mode model is obtained.
        -----------------------------------------------------------------------
        pts        : List of pitch tracks of chunks that belong to the same
                    mode. The pitch distributions of these are iteratively
                    generated to use as the sample points of the mode model
        ref_freq   : Reference frequency to be used in PD/PCD generation.
                    Since this the training function, this should be the
                    annotated tonic of the recording
        metric     : The choice of PCD or PD
        --------------------------------------------------------------------"""
        dist_list = []
        # Iterates over the pitch tracks of a recording
        for idx, _ in enumerate(pts):
            # PitchDistribution of the current chunk is generated
            dist = PitchDistribution.from_cent_pitch(
                pts[idx], ref_freq=ref_freq, smooth_factor=self.smooth_factor,
                step_size=self.step_size)
            if metric == 'pcd':
                dist = dist.to_pcd()

            # The resultant pitch distributions are filled in the list to be
            # returned
            dist_list.append(dist)
        return dist_list
