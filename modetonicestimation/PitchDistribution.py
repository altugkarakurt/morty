# -*- coding: utf-8 -*-
import essentia
import essentia.standard as std
import numpy as np
import json


class PitchDistribution:
    def __init__(self, pd_bins, pd_vals, kernel_width=7.5, ref_freq=440.0):
        """------------------------------------------------------------------------
        The main data structure that wraps all the relevant information about a
        pitch distribution.
        ---------------------------------------------------------------------------
        bins         : Bins of the pitch distribution. It is a 1-D list of equally
                       spaced monotonically increasing frequency values.
        step_size    : The step_size of the distribution bins.
        vals         : Values of the pitch distribution
        ref_freq     : Reference frequency that is used while generating the 
                       distribution. If the tonic of a recording is annotated,
                       this is variable that stores it.
        kernel_width : The std. deviation of the Gaussian kernel. See generate_pd()
                       of ModeFunctions for more detail.
        -------------------------------------------------------------------------"""
        self.bins = np.array(pd_bins)  # force numpy array
        self.vals = np.array(pd_vals)  # force numpy array
        self.kernel_width = kernel_width
        self.ref_freq = np.array(ref_freq)  # force numpy array

        # Due to the floating point issues in Python, the step_size might not be
        # exactly equal to (for example) 7.5, but 7.4999... In such cases the
        # bin generation of pitch distributions include 1200 cents too and chaos
        # reigns. We fix it here.
        temp_ss = self.bins[1] - self.bins[0]
        self.step_size = temp_ss if temp_ss == (round(temp_ss * 10) / 10) else round(temp_ss * 10) / 10

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    @classmethod
    def load(cls, file_name):
        """-------------------------------------------------------------------------
        Loads a PitchDistribution object from JSON file.
        ----------------------------------------------------------------------------
        fname    : The filename of the JSON file
        -------------------------------------------------------------------------"""
        dist = json.load(open(file_name, 'r'))

        return PitchDistribution(dist[0]['bins'], dist[0]['vals'],
                                 kernel_width=dist[0]['kernel_width'],
                                 ref_freq=dist[0]['ref_freq'])

    def save(self, fpath):
        """-------------------------------------------------------------------------
        Saves the PitchDistribution object to a JSON file.
        ----------------------------------------------------------------------------
        fpath    : The file path of the JSON file to be created.
        -------------------------------------------------------------------------"""
        dist_json = [{'bins': self.bins.tolist(), 'vals': self.vals.tolist(),
                      'kernel_width': self.kernel_width, 'ref_freq': self.ref_freq.tolist(),
                      'step_size': self.step_size}]

        json.dump(dist_json, open(fpath, 'w'), indent=4)

    def is_pcd(self):
        """-------------------------------------------------------------------------
        The boolean flag of whether the instance is PCD or not.
        -------------------------------------------------------------------------"""
        return max(self.bins) == (1200 - self.step_size) and min(self.bins) == 0

    def detect_peaks(self):
        """-------------------------------------------------------------------------
        Finds the peak indices of the distribution. These are treated as tonic
        candidates in higher order functions.
        -------------------------------------------------------------------------"""
        # Peak detection is handled by Essentia
        detector = std.PeakDetection()
        peak_bins, peak_vals = detector(essentia.array(self.vals))

        # Essentia normalizes the positions to 1, they are converted here
        # to actual index values to be used in bins.
        peak_idxs = [round(bn * (len(self.bins) - 1)) for bn in peak_bins]
        if peak_idxs[0] == 0:
            peak_idxs = np.delete(peak_idxs, [len(peak_idxs) - 1])
            peak_vals = np.delete(peak_vals, [len(peak_vals) - 1])
        return peak_idxs, peak_vals

    def shift(self, shift_idx):
        """-------------------------------------------------------------------------
        Shifts the distribution by the given number of samples
        ----------------------------------------------------------------------------
        shift_idx : The number of samples that the distribution is tı be shifted
        -------------------------------------------------------------------------"""
        # If the shift index is non-zero, do the shifting procedure
        if shift_idx:

            # If distribution is a PCD, we do a circular shift
            if self.is_pcd():
                shifted_vals = np.concatenate((self.vals[shift_idx:], self.vals[:shift_idx]))

            # If distribution is a PD, it just shifts the values. In this case,
            # pd_zero_pad() of ModeFunctions is always applied beforehand to make
            # sure that no non-zero values are dropped.
            else:
                # Shift towards left
                if shift_idx > 0:
                    shifted_vals = np.concatenate((self.vals[shift_idx:], np.zeros(shift_idx)))

                # Shift towards right
                else:
                    shifted_vals = np.concatenate((np.zeros(abs(shift_idx)), self.vals[:shift_idx]))

            return PitchDistribution(self.bins, shifted_vals, kernel_width=self.kernel_width,
                                     ref_freq=self.ref_freq)

        # If a zero sample shift is requested, a copy of the original distribution
        # is returned
        else:
            return PitchDistribution(self.bins, self.vals, kernel_width=self.kernel_width,
                                     ref_freq=self.ref_freq)
