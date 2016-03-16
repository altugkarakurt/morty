# -*- coding: utf-8 -*-
import essentia
import essentia.standard as std
import numpy as np
import json
from scipy.stats import norm
from scipy.integrate import simps
from Converter import Converter


class PitchDistribution:
    def __init__(self, pd_bins, pd_vals, kernel_width=7.5, ref_freq=440.0):
        """-------------------------------------------------------------------
        The main data structure that wraps all the relevant information about a
        pitch distribution.
        ----------------------------------------------------------------------
        pd_bins      : Bins of the pitch distribution. It is a 1-D list of
                       equally spaced monotonically increasing frequency
                       values.
        pd_vals      : Values of the pitch distribution
        kernel_width : The standard deviation of the Gaussian kernel. See
                       generate_pd() of ModeFunctions for more detail.
        ref_freq     : Reference frequency that is used while generating the
                       distribution. If the tonic of a recording is annotated,
                       this is variable that stores it.
        --------------------------------------------------------------------"""
        self.bins = np.array(pd_bins)  # force numpy array
        self.vals = np.array(pd_vals)  # force numpy array
        self.kernel_width = kernel_width
        self.ref_freq = np.array(ref_freq)  # force numpy array

        # Due to the floating point issues in Python, the step_size might not
        # be exactly equal to (for example) 7.5, but 7.4999... In such cases
        # the bin generation of pitch distributions include 1200 cents too
        # and chaos reigns. We fix it here.
        temp_ss = self.bins[1] - self.bins[0]
        self.step_size = temp_ss if temp_ss == (round(temp_ss * 10) / 10) \
            else round(temp_ss * 10) / 10

    @staticmethod
    def from_cent_pitch(cent_track, ref_freq=440, smooth_factor=7.5,
                        step_size=7.5):
        """--------------------------------------------------------------------
        Given the pitch track in the unit of cents, generates the Pitch
        Distribution of it. the pitch track from a text file. 0th column is the
        time-stamps and
        1st column is the corresponding frequency values.
        -----------------------------------------------------------------------
        cent_track:     1-D array of frequency values in cents.
        ref_freq:       Reference frequency used while converting Hz values to
                        cents.
                        This number isn't used in the computations, but is to
                        be recorded in the PitchDistribution object.
        smooth_factor:  The standard deviation of the gaussian kernel, used in
                        Kernel Density Estimation. If 0, a histogram is given
        step_size:      The step size of the Pitch Distribution bins.
        --------------------------------------------------------------------"""

        # Some extra interval is added to the beginning and end since the
        # superposed Gaussian for smooth_factor would introduce some tails in
        # the ends. These vanish after 3 sigmas(=smooth_factor).

        # The limits are also quantized to be a multiple of chosen step-size
        # smooth_factor = standard deviation of the gaussian kernel

        # TODO: filter out the NaN, -infinity and +infinity from the pitch
        # track

        # Finds the endpoints of the histogram edges. Histogram bins will be
        # generated as the midpoints of these edges.
        min_edge = min(cent_track) - (step_size / 2.0)
        max_edge = max(cent_track) + (step_size / 2.0)
        pd_edges = np.concatenate(
            [np.arange(-step_size / 2.0, min_edge, -step_size)[::-1],
             np.arange(step_size / 2.0, max_edge, step_size)])

        # An exceptional case is when min_bin and max_bin are both positive
        # In this case, pd_edges would be in the range of [step_size/2, max_
        # bin]. If so, a -step_size is inserted to the head, to make sure 0
        # would be in pd_bins. The same procedure is repeated for the case
        # when both are negative. Then, step_size is inserted to the tail.
        pd_edges = pd_edges if -step_size / 2.0 in pd_edges else np.insert(
            pd_edges, 0, -step_size / 2.0)
        pd_edges = pd_edges if step_size / 2.0 in pd_edges else np.append(
            pd_edges, step_size / 2.0)

        # Generates the histogram and bins (i.e. the midpoints of edges)
        pd_vals, pd_edges = np.histogram(cent_track, bins=pd_edges,
                                         density=True)
        pd_bins = np.convolve(pd_edges, [0.5, 0.5])[1:-1]

        if smooth_factor > 0:  # kernel density estimation (approximated)
            # smooth the histogram
            normal_dist = norm(loc=0, scale=smooth_factor)
            xn = np.concatenate(
                [np.arange(0, - 5 * smooth_factor, -step_size)[::-1],
                 np.arange(step_size, 5 * smooth_factor, step_size)])
            sampled_norm = normal_dist.pdf(xn)
            if len(sampled_norm) <= 1:
                raise ValueError("the smoothing factor is too small compared "
                                 "to the step size, such that the convolution "
                                 "kernel returns a single point gaussian. "
                                 "Either increase the value to at least "
                                 "(step size/3) or assign smooth factor to 0, "
                                 "for no smoothing.")

            # convolution generates tails
            extra_num_bins = len(sampled_norm) / 2
            pd_vals = np.convolve(pd_vals,
                                  sampled_norm)[extra_num_bins:-extra_num_bins]

            # normalize the area under the curve
            area = simps(pd_vals, dx=step_size)
            pd_vals = pd_vals / area

        # Sanity check. If the histogram bins and vals lengths are different,
        # we are in trouble. This is an important assumption of higher level
        # functions
        if len(pd_bins) != len(pd_vals):
            raise ValueError('Lengths of bins and Vals are different')

        # Initializes the PitchDistribution object and returns it.
        return PitchDistribution(pd_bins, pd_vals, kernel_width=smooth_factor,
                                 ref_freq=ref_freq)

    @staticmethod
    def from_hz_pitch(hz_track, ref_freq=440, smooth_factor=7.5,
                      step_size=7.5):
        cent_track = Converter.hz_to_cent(hz_track, ref_freq)
        return PitchDistribution.from_cent_pitch(
            cent_track, ref_freq=ref_freq, smooth_factor=smooth_factor,
            step_size=step_size)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    @classmethod
    def load(cls, file_name):
        """--------------------------------------------------------------------
        Loads a PitchDistribution object from JSON file.
        -----------------------------------------------------------------------
        file_name    : The filename of the JSON file
        --------------------------------------------------------------------
        """
        dist = json.load(open(file_name, 'r'))

        return PitchDistribution(dist[0]['bins'], dist[0]['vals'],
                                 kernel_width=dist[0]['kernel_width'],
                                 ref_freq=dist[0]['ref_freq'])

    def save(self, fpath):
        """--------------------------------------------------------------------
        Saves the PitchDistribution object to a JSON file.
        -----------------------------------------------------------------------
        fpath    : The file path of the JSON file to be created.
        --------------------------------------------------------------------"""
        dist_json = [{'bins': self.bins.tolist(), 'vals': self.vals.tolist(),
                      'kernel_width': self.kernel_width,
                      'ref_freq': self.ref_freq.tolist(),
                      'step_size': self.step_size}]

        json.dump(dist_json, open(fpath, 'w'), indent=4)

    def is_pcd(self):
        """--------------------------------------------------------------------
        The boolean flag of whether the instance is PCD or not.
        --------------------------------------------------------------------"""
        return (max(self.bins) == (1200 - self.step_size) and
                min(self.bins) == 0)

    def detect_peaks(self):
        """--------------------------------------------------------------------
        Finds the peak indices of the distribution. These are treated as tonic
        candidates in higher order functions.
        --------------------------------------------------------------------"""
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

    def to_pcd(self):
        """--------------------------------------------------------------------
        Given the pitch distribution of a recording, generates its pitch class
        distribution, by octave wrapping.
        -----------------------------------------------------------------------
        pD: PitchDistribution object. Its attributes include everything we need
        --------------------------------------------------------------------"""
        if self.is_pcd():
            return PitchDistribution(self.bins, self.vals,
                                     kernel_width=self.kernel_width,
                                     ref_freq=self.ref_freq)
        else:
            # Initializations
            pcd_bins = np.arange(0, 1200, self.step_size)
            pcd_vals = np.zeros(len(pcd_bins))

            # Octave wrapping
            for k in range(len(self.bins)):
                idx = int((self.bins[k] % 1200) / self.step_size)
                idx = idx if idx != 160 else 0
                pcd_vals[idx] += self.vals[k]

            # Initializes the PitchDistribution object and returns it.
            return PitchDistribution(pcd_bins, pcd_vals,
                                     kernel_width=self.kernel_width,
                                     ref_freq=self.ref_freq)

    def shift(self, shift_idx):
        """--------------------------------------------------------------------
        Shifts the distribution by the given number of samples
        -----------------------------------------------------------------------
        shift_idx : The number of samples that the distribution is to be
                    shifted
        --------------------------------------------------------------------"""
        # If the shift index is non-zero, do the shifting procedure
        if shift_idx:

            # If distribution is a PCD, we do a circular shift
            if self.is_pcd():
                shifted_vals = np.concatenate((self.vals[shift_idx:],
                                               self.vals[:shift_idx]))

            # If distribution is a PD, it just shifts the values. In this case,
            # pd_zero_pad() of ModeFunctions is always applied beforehand to
            # make sure that no non-zero values are dropped.
            else:
                # Shift towards left
                if shift_idx > 0:
                    shifted_vals = np.concatenate((self.vals[shift_idx:],
                                                   np.zeros(shift_idx)))

                # Shift towards right
                else:
                    shifted_vals = np.concatenate((np.zeros(abs(shift_idx)),
                                                   self.vals[:shift_idx]))

            return PitchDistribution(self.bins, shifted_vals,
                                     kernel_width=self.kernel_width,
                                     ref_freq=self.ref_freq)

        # If a zero sample shift is requested, a copy of the original
        # distribution is returned
        else:
            return PitchDistribution(self.bins, self.vals,
                                     kernel_width=self.kernel_width,
                                     ref_freq=self.ref_freq)
