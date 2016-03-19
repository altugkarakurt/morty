# -*- coding: utf-8 -*-
import essentia
import essentia.standard as std
import numpy as np
import json
from scipy.stats import norm
from scipy.integrate import simps
import matplotlib.pyplot as plt
from Converter import Converter
import numbers


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

    @property
    def bin_unit(self):
        if self.ref_freq is None:
            return 'Hz'
        elif isinstance(self.ref_freq, numbers.Number) and self.ref_freq > 0:
            return 'cent'
        elif self.ref_freq.tolist() and self.ref_freq > 0:  # numpy array
            return 'cent'
        else:
            return ValueError('Invalid reference. ref_freq should be either '
                              'None (bins in Hz) or a number greater than 0.')

    @staticmethod
    def from_cent_pitch(cent_track, ref_freq=440, smooth_factor=7.5,
                        step_size=7.5, norm_type='area'):
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
        # parse the cent_track
        cent_track = np.copy(cent_track)
        if cent_track.ndim > 1:  # pitch is given as [time, pitch, (conf)]
            cent_track = cent_track[:, 1]

        # filter out NaN, and infinity
        cent_track = cent_track[~np.isnan(cent_track)]
        cent_track = cent_track[~np.isinf(cent_track)]

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
                                         density=False)
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
            pd_bins = np.concatenate(
                (np.arange(pd_bins[0] - extra_num_bins * step_size,
                           pd_bins[0], step_size), pd_bins,
                 np.arange(pd_bins[-1] + step_size, pd_bins[-1] +
                           extra_num_bins * step_size + step_size,
                           step_size)))

            pd_vals = np.convolve(pd_vals, sampled_norm)

        # Sanity check. If the histogram bins and vals lengths are different,
        # we are in trouble. This is an important assumption of higher level
        # functions
        if len(pd_bins) != len(pd_vals):
            raise ValueError('Lengths of bins and Vals are different')

        # initialize the distribution
        pd = PitchDistribution(pd_bins, pd_vals, kernel_width=smooth_factor,
                               ref_freq=ref_freq)

        # normalize
        pd.normalize(norm_type=norm_type)

        return pd

    @staticmethod
    def from_hz_pitch(hz_track, ref_freq=440, smooth_factor=7.5,
                      step_size=7.5, norm_type='area'):
        hz_track = np.copy(hz_track)
        if hz_track.ndim > 1:  # pitch is given as [time, pitch, (conf)] array
            hz_track = hz_track[:, 1]

        # filter out the NaN, -infinity and +infinity and values < 20
        hz_track = hz_track[~np.isnan(hz_track)]
        hz_track = hz_track[~np.isinf(hz_track)]
        hz_track = hz_track[hz_track >= 20.0]
        cent_track = Converter.hz_to_cent(hz_track, ref_freq, min_freq=20.0)

        return PitchDistribution.from_cent_pitch(
            cent_track, ref_freq=ref_freq, smooth_factor=smooth_factor,
            step_size=step_size, norm_type=norm_type)

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
        try:
            dist = json.load(open(file_name, 'r'))
        except IOError:  # json string
            dist = json.loads(file_name)

        return PitchDistribution.from_dict(dist)

    @staticmethod
    def from_dict(distrib_dict):
        return PitchDistribution(distrib_dict['bins'], distrib_dict['vals'],
                                 kernel_width=distrib_dict['kernel_width'],
                                 ref_freq=distrib_dict['ref_freq'])

    def to_dict(self):
        pdict = self.__dict__
        for key in pdict.keys():
            try:
                # convert to list from np array
                pdict[key] = pdict[key].tolist()
            except AttributeError:
                pass

        return pdict

    def save(self, fpath=None):
        """--------------------------------------------------------------------
        Saves the PitchDistribution object to a JSON file.
        -----------------------------------------------------------------------
        fpath    : The file path of the JSON file to be created.
        --------------------------------------------------------------------"""
        dist_json = self.to_dict()

        if fpath is None:
            json.dumps(dist_json, indent=4)
        else:
            json.dump(dist_json, open(fpath, 'w'), indent=4)

    def is_pcd(self):
        """--------------------------------------------------------------------
        The boolean flag of whether the instance is PCD or not.
        --------------------------------------------------------------------"""
        return (max(self.bins) == (1200 - self.step_size) and
                min(self.bins) == 0)

    def has_hz_bin(self):
        return self.bin_unit in ['hz', 'Hz', 'Hertz', 'hertz']

    def has_cent_bin(self):
        return self.bin_unit in ['cent', 'Cent', 'cents', 'Cents']

    def normalize(self, norm_type='area'):
        if norm_type is None:  # nothing, keep the occurences (histogram)
            pass
        elif norm_type == 'area':  # area under the curve using simpsons rule
            area = simps(self.vals, dx=self.step_size)
            self.vals /= area
        elif norm_type == 'sum':  # sum normalization
            sumval = np.sum(self.vals)
            self.vals /= sumval
        elif norm_type == 'max':  # max number becomes 1
            maxval = max(self.vals)
            self.vals /= maxval
        else:
            raise ValueError("norm_type can be None, 'area', 'sum' or 'max'")

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
        peak_idxs = [int(round(bn * (len(self.bins) - 1))) for bn in peak_bins]
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

    def hz_to_cent(self, ref_freq):
        if self.has_hz_bin():
            self.bins = Converter.hz_to_cent(self.bins, ref_freq)
            self.ref_freq = ref_freq
        else:
            raise ValueError('The bin unit should be "hz".')

    def cent_to_hz(self):
        if self.has_cent_bin():
            self.bins = Converter.cent_to_hz(self.bins, self.ref_freq)
            self.ref_freq = None
        else:
            raise ValueError('The bin unit should be "cent".')

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

    def plot(self):
        plt.plot(self.bins, self.vals)
        if self.is_pcd():
            plt.title('Pitch class distribution')
        else:
            plt.title('Pitch distribution')
        if self.has_hz_bin():
            plt.xlabel('Frequency (Hz)')
        else:
            plt.xlabel('Normalized Frequency (cent), ref = ' +
                       str(self.ref_freq))
        plt.ylabel('Occurence')
