import numpy as np


class Converter(object):
    @staticmethod
    def hz_to_cent(hz_track, ref_freq):
        """--------------------------------------------------------------------
        Converts an array of Hertz values into cents.
        -----------------------------------------------------------------------
        hz_track : The 1-D array of Hertz values
        ref_freq    : Reference frequency for cent conversion
        --------------------------------------------------------------------"""
        hz_track = np.array(hz_track)

        # The 0 Hz values are removed, not only because they are meaningless,
        # but also logarithm of 0 is problematic.
        return np.log2(hz_track[hz_track > 0] / ref_freq) * 1200.0

    @staticmethod
    def cent_to_hz(cent_track, ref_freq):
        """--------------------------------------------------------------------
        Converts an array of cent values into Hertz.
        -----------------------------------------------------------------------
        cent_track  : The 1-D array of cent values
        ref_freq    : Reference frequency for cent conversion
        --------------------------------------------------------------------"""
        cent_track = np.array(cent_track)

        return 2 ** (cent_track / 1200.0) * ref_freq