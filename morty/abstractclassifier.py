# -*- coding: utf-8 -*-
import numpy as np
from converter import Converter
from pitchdistribution import PitchDistribution
import abc
from scipy.spatial import distance as spdistance
import collections


class AbstractClassifier(object):
    __metaclass__ = abc.ABCMeta
    _estimate_kwargs = abc.abstractproperty
    _dummy_ref_freq = 220.0

    def __init__(self, step_size=7.5, smooth_factor=7.5, feature_type='pcd',
                 models=None):
        """--------------------------------------------------------------------
        These attributes are wrapped as an object since these are used in both
        training and estimation stages and must be consistent in both processes
        -----------------------------------------------------------------------
        step_size       : Step size of the distribution bins
        smooth_factor   : Standart deviation of the gaussian kernel used to
                          smoothen the distributions. For further details,
                          1see generate_pd() of ModeFunctions.
        feature_type    : The feature type to be used in training and testing
                          ("pd" for pitch distribution, "pcd" for pitch
                          class distribution)
        --------------------------------------------------------------------"""
        self.smooth_factor = smooth_factor
        self.step_size = step_size

        assert feature_type in ['pd', 'pcd'], \
            '"feature_type" can either take the value "pd" (pitch ' \
            'distribution) or "pcd" (pitch class distribution).'
        self.feature_type = feature_type

        if models is not None:
            assert all(m['feature'].distrib_type == feature_type
                       for m in models), 'The feature_type input and type ' \
                                         'of the distributions in the ' \
                                         'models input does not match'
        self.models = models

    def _parse_tonic_and_joint_estimate_input(self, test_input):
        if isinstance(test_input, PitchDistribution):  # pitch distribution
            assert test_input.has_hz_bin(), 'The input distribution has a ' \
                                            'reference frequency already.'
            return test_input.hz_to_cent(self._dummy_ref_freq)
        else:  # pitch track or file
            pitch_cent = self._parse_pitch_input(test_input,
                                                 self._dummy_ref_freq)
            return self._cent_pitch_to_feature(pitch_cent,
                                               self._dummy_ref_freq)

    def _parse_mode_estimate_input(self, *args):
        assert len(args) < 3, 'Too many inputs.'

        if isinstance(args[0], PitchDistribution):
            feature = args[0]
            if len(args) == 2:  # tonic given
                feature.hz_to_cent(args[1])
        else:  # pitch
            if len(args) == 1:  # pitch given in cent units
                pitch_cent = args[0]
                ref_freq = self._dummy_ref_freq
            else:  # tonic given. note: length can only be 2 since we have
                # the input length assertion in the first line of the method
                pitch_cent = self._parse_pitch_input(*args)
                ref_freq = args[1]
            feature = self._cent_pitch_to_feature(pitch_cent, ref_freq)

        return feature

    @staticmethod
    def _parse_pitch_input(pitch_in, tonic_freq):
        """
        Parses the pitch input from list, numpy array or file.

        If the input (or the file content) is a matrix, the method assumes the
        columns represent timestamps, pitch and "other columns".
        respectively. It only returns the second column in this case.

        :param pitch_in: pitch input, which is a list, numpy array or filename
        :param tonic_freq: the tonic frequency in Hz
        :return: parsed pitch track (numpy array)
        """
        # parse the pitch track from txt file, list or numpy array
        p = np.loadtxt(pitch_in)  # loadtxt converts lists to np.array too
        p = p[:, 1] if p.ndim > 1 else p  # get the pitch stream

        # normalize wrt tonic
        return Converter.hz_to_cent(p, tonic_freq)

    def _cent_pitch_to_feature(self, pitch_cent, ref_freq):
        feature = PitchDistribution.from_cent_pitch(
            pitch_cent, ref_freq=ref_freq, smooth_factor=self.smooth_factor,
            step_size=self.step_size)
        if self.feature_type == 'pcd':
            feature = feature.to_pcd()

        return feature

    def _estimate(self, test_feature, mode=None, est_tonic=True,
                  distance_method='bhat', k_param=1, rank=1):
        assert est_tonic or mode is None, 'Nothing to estimate.'

        if est_tonic is True:
            # find the tonic candidates of the input feature
            test_feature, stable_pitches, peak_idx = self.\
                _get_tonic_candidates(test_feature)
        else:
            # dummy assign the first index
            stable_pitches = np.array([test_feature.ref_freq])
            peak_idx = np.array([0])

        training_features, training_modes = self._get_training_models(mode)

        dist_mat = self._generate_distance_matrix(
            test_feature, peak_idx, training_features,
            distance_method=distance_method)

        # sort results
        sorted_idx = np.argsort(dist_mat, axis=None)
        sorted_stable_pitch_idx, sorted_mode_idx = np.unravel_index(
            sorted_idx, dist_mat.shape)

        # convert from sorted index to sorted tonic frequency and mode
        sorted_tonics = stable_pitches[sorted_stable_pitch_idx]
        sorted_modes = training_modes[sorted_mode_idx]
        sorted_pairs = [(t, m) for t, m in zip(sorted_tonics, sorted_modes)]

        # compute ranked estimations
        ranked_pairs = []
        for r in range(rank):
            cand_pairs = self._get_nearest_neighbors(sorted_pairs, k_param)
            estimation, sorted_pairs = self._select_nearest_neighbor(
                cand_pairs, sorted_pairs)
            ranked_pairs.append(estimation)

        return ranked_pairs

    @staticmethod
    def _get_tonic_candidates(test_feature):
        # find the global minima and shift the distribution there so
        # peak detection does not fail locate a peak in the boundary in
        # octave-wrapped features. For features that are not
        # octave-wrapped this step is harmless.
        global_minima_idx = np.argmin(test_feature.vals)
        shift_feature = test_feature.shift(global_minima_idx)

        # get the peaks of the feature as the tonic candidate indices and
        # compute the stable frequencies from the peak indices
        peak_idx = shift_feature.detect_peaks()[0]
        peaks_cent = shift_feature.bins[peak_idx]
        freqs = Converter.cent_to_hz(peaks_cent, shift_feature.ref_freq)

        # return the shifted feature, stable frequencies and their
        # corresponding index in the shifted feature
        return shift_feature, freqs, peak_idx

    def _get_training_models(self, mode):
        if mode is None:
            training_features = [m['feature'] for m in self.models]
            feature_modes = np.array([m['mode'] for m in self.models])
        else:
            training_features = [m['feature'] for m in self.models
                                 if m['mode'] == mode]
            # create dummy array with annotated mode
            feature_modes = np.array(
                [mode for _ in range(len(training_features))])
        return training_features, feature_modes

    @staticmethod
    def _get_nearest_neighbors(sorted_pair, k_param):
        # parse mode/tonic pairs
        pairs = [(t, m) for t, m in sorted_pair[:k_param]]

        # find the most common pairs
        counter = collections.Counter(pairs)
        most_commons = counter.most_common(k_param)
        max_cnt = most_commons[0][1]
        cand_pairs = [c[0] for c in most_commons if c[1] == max_cnt]

        return cand_pairs

    @staticmethod
    def _select_nearest_neighbor(cand_pairs, sorted_pair):
        # in case there are multiple candidates get the pair sorted earlier
        for p in sorted_pair:
            if p in cand_pairs:
                estimated_pair = p

                # pop the estimated pair from the sorte_pair list for ranking
                sorted_pair = [pp for pp in sorted_pair if pp != p]

                return estimated_pair, sorted_pair

        assert False, 'No pair selected, this should be impossible!'

    @classmethod
    def _chk_estimate_kwargs(cls, **kwargs):
        assert all(key in cls._estimate_kwargs for key in kwargs.keys()), \
            'The input arguments are %s' % ', '.join(kwargs.keys())

    @classmethod
    def _generate_distance_matrix(cls, distrib, peak_idxs, training_distribs,
                                  distance_method='bhat'):
        """--------------------------------------------------------------------
        Iteratively calculates the distance of the input distribution from each
        (mode candidate, tonic candidate) pair. This is a generic function,
        that is independent of distribution type or any other parameter value.
        -----------------------------------------------------------------------
        distribs            : Input distribution that is to be estimated
        peak_idxs           : List of indices of distribution peaks
        training_distribs   : List of training distributions
        method              : The distance method to be used. The available
                              distances are listed in distance() function.
        --------------------------------------------------------------------"""

        result = np.zeros((len(peak_idxs), len(training_distribs)))

        # Iterates over the peaks, i.e. the tonic candidates
        for i, cur_peak_idx in enumerate(peak_idxs):
            trial = distrib.shift(cur_peak_idx)

            # Iterates over mode candidates
            for j, td in enumerate(training_distribs):
                assert trial.bin_unit == td.bin_unit, \
                    'The bin units of the compared distributions should match.'
                assert trial.distrib_type() == td.distrib_type(), \
                    'The features should be of the same type'

                if trial.distrib_type() == 'pd':
                    # compare in the overlapping region
                    min_td_bin = np.min(td.bins)
                    max_td_bin = np.max(td.bins)

                    min_trial_bin = np.min(trial.bins)
                    max_trial_bin = np.max(trial.bins)

                    overlap = [max([min_td_bin, min_trial_bin]),
                               min([max_td_bin, max_trial_bin])]

                    trial_bool = (overlap[0] <= trial.bins) * \
                                 (trial.bins <= overlap[1])
                    trial_vals = trial.vals[trial_bool]

                    td_bool = (overlap[0] <= td.bins) * \
                              (td.bins <= overlap[1])
                    td_vals = td.vals[td_bool]
                else:
                    trial_vals = trial.vals
                    td_vals = td.vals

                # Calls the distance function for each entry of the matrix
                result[i][j] = cls._distance(trial_vals, td_vals,
                                             method=distance_method)
        return np.array(result)

    @staticmethod
    def _distance(vals_1, vals_2, method='bhat'):
        """--------------------------------------------------------------------
         Calculates the distance between two 1-D lists of values. This
         function is called with pitch distribution values, while generating
         distance matrices. The function is symmetric, the two inpıt lists
         are interchangable.
         ----------------------------------------------------------------------
         vals_1, vals_2 : The input value lists.
         method         : The choice of distance method
         ----------------------------------------------------------------------
         manhattan    : Minkowski distance of 1st degree
         euclidean    : Minkowski distance of 2nd degree
         l3           : Minkowski distance of 3rd degree
         bhat         : Bhattacharyya distance
         intersection : Intersection
         corr         : Correlation
         -------------------------------------------------------------------"""
        if method == 'euclidean':
            return spdistance.euclidean(vals_1, vals_2)
        elif method == 'manhattan':
            return spdistance.minkowski(vals_1, vals_2, 1)
        elif method == 'l3':
            return spdistance.minkowski(vals_1, vals_2, 3)
        elif method == 'bhat':
            return -np.log(sum(np.sqrt(vals_1 * vals_2)))
        # Since correlation and intersection are actually similarity measures,
        # we take their inverse to be able to use them as distances. In other
        # words, max. similarity would give the min. inverse and we are always
        # looking for minimum distances.
        elif method == 'intersection':
            return len(vals_1) / (sum(np.minimum(vals_1, vals_2)))
        elif method == 'corr':
            return 1.0 - np.correlate(vals_1, vals_2)
        else:
            return 0
