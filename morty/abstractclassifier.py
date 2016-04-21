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

    @staticmethod
    def _parse_pitch(pitch_in, tonic_freq):
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

    def _cent_pitch_to_feature(self, pitch_cent):
        feature = PitchDistribution.from_cent_pitch(
            pitch_cent, smooth_factor=self.smooth_factor,
            step_size=self.step_size)
        if self.feature_type == 'pcd':
            feature = feature.to_pcd()

        return feature

    def _estimate(self, feature, mode=None, est_tonic=True,
                  distance_method='bhat', k_param=1, rank=1):
        assert est_tonic or mode is None, 'Nothing to estimate.'

        peak_idx = self._get_tonic_candidates(est_tonic)
        features, feature_modes = self._get_model_candidates(mode)

        dist_mat = self._generate_distance_matrix(
            feature, peak_idx, features, distance_method=distance_method)

        # sort results
        sorted_idx = np.argsort(dist_mat, axis=None)
        sorted_peak_idx, sorted_mode_idx = np.unravel_index(
            sorted_idx, dist_mat.shape)

        # convert from index to mode and tonic
        sorted_tonics = sorted_peak_idx  # TODO: convert to tonic frequency
        sorted_modes = feature_modes[sorted_mode_idx]
        sorted_pair = [(t, m) for t, m in zip(sorted_tonics, sorted_modes)]

        # compute ranked estimations
        ranked_pairs = []
        for r in range(rank):
            cand_pairs = self._knn(sorted_pair, k_param)
            estimation, sorted_pair = self._break_knn_tie(
                cand_pairs, sorted_pair)
            ranked_pairs.append(estimation)

        return ranked_pairs

    def _get_model_candidates(self, mode):
        if mode is None:
            features = [m['feature'] for m in self.models]
            feature_modes = np.array([m['mode'] for m in self.models])
        else:
            features = [m['feature'] for m in self.models
                             if m['mode'] == mode]
            # create dummy array with annotated mode
            feature_modes = np.array([mode for _ in range(len(features))])
        return features, feature_modes

    def _get_tonic_candidates(self, est_tonic):
        if est_tonic is True:  # find the tonic candidates of the input feature
            # TODO: implement tonic identification
            raise NotImplementedError
        else:
            # only try for the first feature index
            peak_idx = [0]
        return peak_idx

    @staticmethod
    def _break_knn_tie(cand_pairs, sorted_pair):
        # in case there are multiple candidates get the pair sorted earlier
        for p in sorted_pair:
            if p in cand_pairs:
                estimated_pair = p

                # pop the estimated pair from the sorte_pair list for ranking
                sorted_pair = [pp for pp in sorted_pair if pp != p]

                return estimated_pair, sorted_pair

        assert False, 'No pair selected, this should be impossible!'

    @staticmethod
    def _knn(sorted_pair, k_param):
        # parse mode/tonic pairs
        pairs = [(t, m) for t, m in sorted_pair[:k_param]]

        # find the most common pairs
        counter = collections.Counter(pairs)
        most_commons = counter.most_common(k_param)
        max_cnt = most_commons[0][1]
        cand_pairs = [c[0] for c in most_commons if c[1] == max_cnt]

        return cand_pairs

    def _parse_estimate_args(self, *args):
        if isinstance(args, PitchDistribution):  # precomputed pitch distrib.
            feature = args
            try:
                assert self.feature_type == feature.distrib_type(), \
                    'The training feature_types and type of the input ' \
                    'distribution does not match'
            except AttributeError:  # normalized pitch track
                feature = self._cent_pitch_to_feature(args)
        elif len(args) == 2:  # (pitch, tonic)
            pitch_cent = self._parse_pitch(*args)
            feature = self._cent_pitch_to_feature(pitch_cent)
        else:
            raise ValueError("The input can be either ")

        return feature

    @classmethod
    def _chk_estimate_kwargs(cls, **kwargs):
        assert all(key in cls._estimate_kwargs for key in kwargs.keys()), \
            'The input arguments are %s' % ', '.join(kwargs.keys())

    @classmethod
    def _generate_distance_matrix(cls, distrib, peak_idxs, mode_distribs,
                                  distance_method='bhat'):
        """--------------------------------------------------------------------
        Iteratively calculates the distance of the input distribution from each
        (mode candidate, tonic candidate) pair. This is a generic function,
        that is independent of distribution type or any other parameter value.
        -----------------------------------------------------------------------
        distribs        : Input distribution that is to be estimated
        peak_idxs       : List of indices of distribution peaks
        mode_distribs   : List of candidate mode distributions
        method          : The distance method to be used. The available
                          distances are listed in distance() function.
        --------------------------------------------------------------------"""

        result = np.zeros((len(peak_idxs), len(mode_distribs)))

        # Iterates over the peaks, i.e. the tonic candidates
        for i, cur_peak_idx in enumerate(peak_idxs):
            trial = distrib.shift(cur_peak_idx)

            # Iterates over mode candidates
            for j, cur_mode_dist in enumerate(mode_distribs):
                # Calls the distance function for each entry of the matrix
                result[i][j] = cls._distance(trial.vals, cur_mode_dist.vals,
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
