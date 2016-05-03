# -*- coding: utf-8 -*-

import numpy as np

from .classifierinputparser import ClassifierInputParser
from .knn import KNN
from ..converter import Converter
from ..pitchdistribution import PitchDistribution


class KNNClassifier(ClassifierInputParser):
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
        models          : Pre-trained models per mode
        --------------------------------------------------------------------"""
        super(KNNClassifier, self).__init__(
            step_size=step_size, smooth_factor=smooth_factor,
            feature_type=feature_type, models=models)

    def train(self, pitches, tonics, modes, sources=None, model_type='multi'):
        if model_type=='single':
            return self._train_single_distrib__per_mode(
                pitches, tonics, modes, sources=sources)
        elif model_type == 'multi':
            return self._train_single_distrib__per_mode(
                pitches, tonics, modes, sources=sources)
        else:
            raise ValueError("Unknown training model")

    def _train_single_distrib__per_mode(self, pitches, tonics, modes,
                                        sources=None):
        """--------------------------------------------------------------------
        For the mode trainings, the requirements are a set of recordings with
        annotated tonics for each mode under consideration. This function only
        expects the recordings' pitch tracks and corresponding tonics as lists.
        The two lists should be indexed in parallel, so the tonic of ith pitch
        track in the pitch track list should be the ith element of tonic list.
        Once training is completed for a mode, the model would be generated
        as a PitchDistribution object and saved in a JSON file. For loading
        these objects and other relevant information about the data structure,
        see the PitchDistribution class.
        -----------------------------------------------------------------------
        pitches       : List of pitch tracks or the list of files with
                        stored pitch tracks (i.e. single-column
                        lists/numpy arrays/files with frequencies)
        tonics        : List of annotated tonic frequencies of recordings
        modes         : Name of the modes of each training sample.
        --------------------------------------------------------------------"""

        assert len(pitches) == len(modes) == len(tonics), \
            'The inputs should have the same length!'

        # get the pitch tracks for each mode and convert them to cent unit
        tmp_models = {m: {'sources': [], 'cent_pitch': []} for m in
                      set(modes)}
        for p, t, m, s in zip(pitches, tonics, modes, sources):
            # parse the pitch track from txt file, list or numpy array and
            # normalize with respect to annotated tonic
            pitch_cent = self._parse_pitch_input(p, t)

            # convert to cent track and append to the mode data
            tmp_models[m]['cent_pitch'].extend(pitch_cent)
            tmp_models[m]['sources'].append(s)

        # compute the feature for each model from the normalized pitch tracks
        for model in tmp_models.values():
            model['feature'] = PitchDistribution.from_cent_pitch(
                model.pop('cent_pitch', None),
                smooth_factor=self.smooth_factor, step_size=self.step_size)

            # convert to pitch-class distribution if requested
            if self.feature_type == 'pcd':
                model['feature'] = model['feature'].to_pcd()

        # make the models a list of dictionaries by collapsing the mode keys
        # inside the values
        models = []
        for mode_name, model in tmp_models.items():
            model['mode'] = mode_name
            models.append(model)

        self.models = models

    def _train_multi_distrib_per_mode(self, pitches, tonics, modes,
                                      sources=None):
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
        feature       : Whether the model should be octave wrapped (Pitch Class
                        Distribution: PCD) or not (Pitch Distribution: PD)
        save_dir      : Where to save the resultant JSON files.
        --------------------------------------------------------------------"""
        assert len(pitches) == len(modes) == len(tonics), \
            'The inputs should have the same length!'

        # get the pitch tracks for each mode and convert them to cent unit
        models = []
        for p, t, m, s in zip(pitches, tonics, modes, sources):
            # parse the pitch track from txt file, list or numpy array and
            # normalize with respect to annotated tonic
            pitch_cent = self._parse_pitch_input(p, t)
            feature = PitchDistribution.from_cent_pitch(
                pitch_cent, smooth_factor=self.smooth_factor,
                step_size=self.step_size)

            # convert to pitch-class distribution if requested
            if self.feature_type == 'pcd':
                feature = feature.to_pcd()

            model = {'source': s, 'tonic': t, 'mode': m,
                     'feature': feature}
            # convert to cent track and append to the mode data
            models.append(model)

        self.models = models

    def identify_tonic(self, test_input, mode, distance_method='bhat',
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

    def estimate_tonic(self, test_input, mode, distance_method='bhat',
                       rank=1):
        """
        Alias of "identify_tonic" method. See the documentation of
        "identify_tonic" for more information.
        """
        return self.identify_tonic(
            test_input, mode, distance_method=distance_method, rank=rank)

    def recognize_mode(self, feature_in, tonic=None, distance_method='bhat',
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

    def estimate_mode(self, feature_in, tonic=None, distance_method='bhat',
                      rank=1):

        return self.recognize_mode(
            feature_in, tonic=tonic, distance_method=distance_method,
            rank=rank)

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
