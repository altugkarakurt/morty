# -*- coding: utf-8 -*-
from pitchdistribution import PitchDistribution
from abstractclassifier import AbstractClassifier


class Bozkurt(AbstractClassifier):
    """------------------------------------------------------------------------
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

    We require a set of recordings with annotated modes and tonics to train
    the mode models. Then, the unknown mode and/or tonic of an input
    recording is estimated by comparing it to these models.

    There are two functions and as their names suggest, one handles the
    training tasks and the other does the estimation once the trainings are
    completed.
    ------------------------------------------------------------------------"""
    _estimate_kwargs = ['distance_method', 'rank']

    def __init__(self, step_size=7.5, smooth_factor=7.5, feature_type='pcd',
                 models=None):
        super(Bozkurt, self).__init__(
            step_size=step_size, smooth_factor=smooth_factor,
            feature_type=feature_type, models=models)

    def train(self, pitches, tonics, modes, sources=None):
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
        tmp_models = {m: {'sources': [], 'cent_pitch': []} for m in set(modes)}
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

    def estimate_mode(self, *args, **kwargs):
        """--------------------------------------------------------------------
        Mode recognition: The tonic of the recording is known and the mode is
        to be estimated.
        :param args: - precomputed feature (PD or PCD in cents)
                     - normalized pitch track (in cents)
                     - precomputed feature (PD or PCD in Hz), tonic frequency
                     - pitch track (in Hz), tonic frequency
        :param kwargs: Input parameters. Possible keys are:
                     - distance_method: distance used in KNN
                     - rank: number of estimations to return
        :return: ranked mode estimations
        --------------------------------------------------------------------"""
        test_feature = self._parse_mode_estimate_input(*args)
        self._chk_estimate_kwargs(**kwargs)

        # Mode Estimation
        estimations = self._estimate(
            test_feature, est_tonic=False, mode=None, **kwargs)

        # remove the dummy tonic estimation
        modes_ranked = [e[1] for e in estimations]

        return modes_ranked

    def estimate_tonic(self, test_input, mode, distance_method='bhat', rank=1):
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
        test_feature = self._parse_tonic_and_joint_estimate_input(test_input)

        # Mode Estimation
        estimations = self._estimate(
            test_feature, est_tonic=True, mode=mode,
            distance_method=distance_method, rank=rank)

        # remove the dummy tonic estimation
        tonics_ranked = [e[0] for e in estimations]

        return tonics_ranked

    def estimate_joint(self, test_input, distance_method='bhat', rank=1):
        """--------------------------------------------------------------------
        Joint estimation: Estimate both the tonic and mode together
        :param test_input: - precomputed feature (PD or PCD in Hz)
                           - pitch track in Hz (list or numpy array)
        :param distance_method: distance used in KNN
        :param rank: number of estimations to return
        :return: ranked mode and tonic estimations
        --------------------------------------------------------------------"""
        test_feature = self._parse_tonic_and_joint_estimate_input(test_input)

        # Mode Estimation
        joint_estimations = self._estimate(
            test_feature, est_tonic=True, mode=None,
            distance_method=distance_method, rank=rank)

        return joint_estimations
