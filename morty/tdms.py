# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy
import scipy.ndimage as ndimage


class TDMS(object):
    def __init__(self, pd_bins, pd_vals, kernel_width=7.5, ref_freq=440.0):
        """-------------------------------------------------------------------
        Time Delayed Melody Surface data structure. As described in
        Sankalp Gulati, et al.: Time-Delayed Melody Surfaces for Rāga
        Recognition. ISMIR 2016
        ----------------------------------------------------------------------
        pd_bins      : Bins of the melody surface. This is a 1D list and is
                       used as support of both dimensions of the 2D feature
        pd_vals      : Height of the surface, a 2D matrix
        kernel_width : The standard deviation of the Gaussian kernel used
        ref_freq     : Reference frequency that is used while generating the
                       distribution. If the tonic of a recording is annotated,
                       this is variable that stores it.
        --------------------------------------------------------------------"""
        self.bins = np.array(pd_bins)  # force numpy array
        self.vals = np.array(pd_vals)  # force numpy array
        self.kernel_width = kernel_width
        if ref_freq is None:
            self.ref_freq = None
        else:
            self.ref_freq = np.array(ref_freq)  # force numpy array

        # get step_size to one decimal point
        temp_ss = self.bins[1] - self.bins[0]
        self.step_size = temp_ss if temp_ss == (round(temp_ss * 10) / 10) \
            else round(temp_ss * 10) / 10

    @staticmethod
    def from_cent_pitch(cent_track, tau, ref_freq=440.0, kernel_width=7.5,
                        step_size=7.5, alpha=1, norm_type="sum", sampling_freq=44100):
        """--------------------------------------------------------------------
        TODO
        -----------------------------------------------------------------------
        cent_track:     1-D array of frequency values in cents.
        tau:            Time-delay index (in seconds)
        ref_freq:       Reference frequency used while converting Hz values to
                        cents.
                        This number isn't used in the computations, but is to
                        be recorded in the PitchDistribution object.
        kernel_width:   The standard deviation of the gaussian kernel, used in
                        Kernel Density Estimation. If 0, a histogram is given
        step_size:      The step size of the Pitch Distribution bins.
        sampling_freq:  Sampling frequency used while recording the track
        --------------------------------------------------------------------"""
        assert ((step_size > 0) and (1200.0 % step_size == 0)), \
            'The step size should have a positive value'

        delay = int(tau * sampling_freq)
        eta = int(1200.0 / step_size)

        # Handling B(x)
        x = np.floor(np.mod((eta / 1200) * cent_track, 1200 * np.ones_like(cent_track)))

        x1 = deepcopy(x[delay:len(x)])
        x2 = deepcopy(x[0:len(x) - delay])
        surface = np.array([[sum(np.logical_and((x1 == i), (x2 == j)))
                            for i in range(eta)] for j in range(eta)])

        # Power compression
        surface = 1.0 * np.power(surface, alpha)

        # Gaussian smoothing
        surface = ndimage.gaussian_filter(surface, kernel_width)

        # Normalization
        if norm_type == 'sum':  # sum normalization
            normval = np.sum(surface)
        elif norm_type == 'max':  # max number becomes 1
            normval = max(surface)
        surface /= normval
