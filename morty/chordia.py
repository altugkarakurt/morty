# -*- coding: utf-8 -*-
from morty.pitchdistribution import PitchDistribution
from abstractclassifier import AbstractClassifier


class Chordia(AbstractClassifier):
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
    _estimate_kwargs = ['distance_method', 'rank', 'chunk_size', 'overlap',
                        'threshold', 'frame_rate', 'k_param']

    def __init__(self, step_size=7.5, smooth_factor=7.5, chunk_size=60,
                 threshold=0.5, overlap=0, frame_rate=128.0 / 44100,
                 feature_type='pcd', models=None):
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
        super(Chordia, self).__init__(
            step_size=step_size, smooth_factor=smooth_factor,
            feature_type=feature_type, models=models)
        self.overlap = overlap
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.frame_rate = frame_rate

    def train(self, pitches, tonics, modes, sources=None):
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

            model = {'source': s, 'tonic': t, 'mode': m, 'feature': feature}
            # convert to cent track and append to the mode data
            models.append(model)

        self.models = models
