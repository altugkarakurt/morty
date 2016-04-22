# -*- coding: utf-8 -*-
import abc

import numpy as np

from classifierinputparser import ClassifierInputParser
from morty.classifiers.knn import KNN
from morty.converter import Converter


class AbstractClassifier(ClassifierInputParser):
    __metaclass__ = abc.ABCMeta
    _input_parser = ClassifierInputParser

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
        super(AbstractClassifier, self).__init__(
            step_size=step_size, smooth_factor=smooth_factor,
            feature_type=feature_type, models=models)

    def estimate_tonic(self, test_input, mode, distance_method='bhat',
                       rank=1):
        """--------------------------------------------------------------------
        Tonic Identification: The mode of the recording is known and the
        tonic is to be estimated.
        :param test_input: - precomputed feature (PD or PCD in Hz)
                           - pitch track in Hz (list or numpy array)
        :param mode: input mode label
        :param distance_method: distance used in KNN
        :param rank: number of estimations to return
        :return: ranked mode estimations
        --------------------------------------------------------------------"""
        test_feature = self._parse_tonic_and_joint_estimate_input(
            test_input)

        # Mode Estimation
        estimations = self._estimate(
            test_feature, est_tonic=True, mode=mode,
            distance_method=distance_method, rank=rank)

        # remove the dummy tonic estimation
        tonics_ranked = [(e[0][0], e[1]) for e in estimations]

        return tonics_ranked

    def estimate_mode(self, feature_in, tonic=None, distance_method='bhat',
                      rank=1):
        """--------------------------------------------------------------------
        Mode recognition: The tonic of the recording is known and the mode is
        to be estimated.
        :param feature_in: - precomputed feature (PitchDistribution object)
                           - pitch track (list or noumpy array)
        :param tonic: tonic frequency (float). It is needed if the feature_in
                      has not been normalized with respect to the tonic earlier
        :param distance_method: distance used in KNN
        :param rank: number of estimations to return
        :return: ranked mode estimations
        --------------------------------------------------------------------"""
        test_feature = self._parse_mode_estimate_input(feature_in, tonic)

        # Mode Estimation
        estimations = self._estimate(
            test_feature, est_tonic=False, mode=None,
            distance_method=distance_method, rank=rank)

        # remove the dummy tonic estimation
        modes_ranked = [(e[0][1], e[1]) for e in estimations]

        return modes_ranked

    def estimate_joint(self, test_input, distance_method='bhat', rank=1):
        """--------------------------------------------------------------------
        Joint estimation: Estimate both the tonic and mode together
        :param test_input: - precomputed feature (PD or PCD in Hz)
                           - pitch track in Hz (list or numpy array)
        :param distance_method: distance used in KNN
        :param rank: number of estimations to return
        :return: ranked mode and tonic estimations
        --------------------------------------------------------------------"""
        test_feature = self._parse_tonic_and_joint_estimate_input(
            test_input)

        # Mode Estimation
        joint_estimations = self._estimate(
            test_feature, est_tonic=True, mode=None,
            distance_method=distance_method, rank=rank)

        return joint_estimations

    def _estimate(self, test_feature, mode=None, est_tonic=True,
                  distance_method='bhat', k_param=1, rank=1):
        assert est_tonic or mode is None, 'Nothing to estimate.'

        if est_tonic is True:
            # find the tonic candidates of the input feature
            test_feature, tonic_cands, peak_idx = self.\
                _get_tonic_candidates(test_feature)
        else:
            # dummy assign the first index
            tonic_cands = np.array([test_feature.ref_freq])
            peak_idx = np.array([0])

        training_features, training_modes = self._get_training_models(mode)
        dist_mat = KNN.generate_distance_matrix(
            test_feature, peak_idx, training_features,
            distance_method=distance_method)

        # sort results
        sorted_idx = np.argsort(dist_mat, axis=None)
        sorted_dists = np.sort(dist_mat, axis=None)
        sorted_tonic_cand_idx, sorted_mode_idx = np.unravel_index(
            sorted_idx, dist_mat.shape)

        # convert from sorted index to sorted tonic frequency and mode
        sorted_tonics = tonic_cands[sorted_tonic_cand_idx]
        sorted_modes = training_modes[sorted_mode_idx]
        sorted_pairs = [((t, m), d) for t, m, d in
                        zip(sorted_tonics, sorted_modes, sorted_dists)]

        # compute ranked estimations
        ranked_pairs = []
        for r in range(rank):
            cand_pairs = KNN.get_nearest_neighbors(sorted_pairs, k_param)
            estimation, sorted_pairs = KNN.select_nearest_neighbor(
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
