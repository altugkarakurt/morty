from morty.neuralnet import NeuralNet
from morty.converter import Converter
from morty.pitchdistribution import PitchDistribution
import numpy as np
import os


class NeuralClassifier(object):
    def __init__(self, step_size, kernel_width, mode_names, weights=None):
        sizes = [int(1200 / step_size), len(mode_names)]
        self.neural_net = NeuralNet(sizes=sizes, weights=weights)
        self.step_size = step_size
        self.kernel_width = kernel_width
        self.mode_dict = dict()
        for idx, mode in enumerate(mode_names):
            self.mode_dict[mode] = [1 if(i == idx) else 0
                                    for i in range(len(mode_names))]

    def train(self, mode_labels, pitch_files, tonic_freqs, learn_rate,
              save_dir=''):
        tonic_freqs = [np.array(tonic) for tonic in tonic_freqs]
        pitch_tracks = self.parse_pitch_input(pitch_files, multiple=True)
        cent_tracks = [Converter.hz_to_cent(k, ref_freq=tonic_freqs[idx])
                       for idx, k in enumerate(pitch_tracks)]
        pcd_vals = [PitchDistribution.from_cent_pitch(
            track, tonic_freqs[idx], self.kernel_width, self.step_size).
            to_pcd().vals for idx, track in enumerate(cent_tracks)]
        self.train_from_pcds(mode_labels, pcd_vals, learn_rate, save_dir)

    def train_from_pcds(self, mode_labels, pcd_list, learn_rate, save_dir=""):
        labels = [self.mode_dict[mode] for mode in mode_labels]
        self.neural_net.train(pcd_list, labels, learn_rate=learn_rate)

        if save_dir:
            fpath = os.path.join(save_dir, "weights.txt")
            np.savetxt(fpath, self.neural_net.weights)

    def mode_estimate(self, pcd_vals):
        mode_est = self.neural_net.estimate(pcd_vals)
        max_idx = np.argmax(mode_est)
        return [1 if(idx == max_idx) else 0 for idx, _ in enumerate(mode_est)]

    @staticmethod
    def parse_pitch_input(pitch_in, multiple=False):
        """--------------------------------------------------------------------
        This function parses all types of pitch inputs that are fed into
        neural network. It can parse inputs of the following form:
        * A (list of) pitch track list(s)
        * A (list of) filename(s) for pitch track(s)
        * A (list of) PitchDistribution object(s)

        This is where we do the type checking to provide the correct input
        format to higher order functions. It returns to things, the cleaned
        input pitch track or pitch distribution.
        --------------------------------------------------------------------"""
        # Multiple pitch tracks
        if multiple:
            return [NeuralClassifier.parse_pitch_input(track, multiple=False)
                    for track in pitch_in]

        # Single pitch track
        else:
            # Path to the pitch track
            if (type(pitch_in) == str) or (type(pitch_in) == unicode):
                if os.path.exists(pitch_in):
                    result = np.loadtxt(pitch_in)
                    # Strip the time track
                    return result[:, 1] if result.ndim > 1 else result
                # Non-path string
                else:
                    raise ValueError("Path doesn't exist: " + pitch_in)

            # Loaded pitch track passed
            elif (type(pitch_in) == np.ndarray) or (type(pitch_in) == list):
                return np.array(pitch_in)
