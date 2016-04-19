from NeuralNet import NeuralNet
from Converter import Converter
import ModeFunctions as ModeFun
from PitchDistribution import PitchDistribution
import numpy as np
import os


class NeuralClassifier:
    def __init__(self, step_size, smooth_factor, mode_names, weights=None):
        sizes = [int(1200 / step_size), len(mode_names)]
        self.neural_net = NeuralNet(sizes=sizes, weights=weights)
        self.step_size = step_size
        self.smooth_factor = smooth_factor
        self.mode_dict = dict()
        for idx, mode in enumerate(mode_names):
            self.mode_dict[mode] = [1 if(i == idx) else 0
                                    for i in range(len(mode_names))]

    def train(self, mode_labels, pitch_files, tonic_freqs, learn_rate,
              save_dir=''):
        tonic_freqs = [np.array(tonic) for tonic in tonic_freqs]
        pitch_tracks = ModeFun.parse_pitch_track(pitch_files, multiple=True)
        cent_tracks = [Converter.hz_to_cent(k, ref_freq=tonic_freqs[idx])
                       for idx, k in enumerate(pitch_tracks)]
        pcd_vals = [PitchDistribution.from_cent_pitch(track,
                    tonic_freqs[idx], self.smooth_factor,
                    self.step_size).to_pcd().vals
                    for idx, track in enumerate(cent_tracks)]
        self.train_from_pcds(mode_labels, pcd_vals, learn_rate, save_dir)

    def train_from_pcds(self, mode_labels, pcd_list, learn_rate, save_dir=""):
        labels = [self.mode_dict[mode] for mode in mode_labels]
        self.neural_net.train(pcd_list, labels, learn_rate=learn_rate)

        if(save_dir):
            fpath = os.path.join(save_dir, "weights.txt")
            np.savetxt(fpath, self.neural_net.weights)

    def mode_estimate(self, pcd_vals):
        mode_est = self.neural_net.estimate(pcd_vals)
        max_idx = np.argmax(mode_est)
        return [1 if(idx == max_idx) else 0 for idx, _ in enumerate(mode_est)]
