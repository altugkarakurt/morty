# -*- coding: utf-8 -*-
import numpy as np
import os


def parse_pitch_track(pitch_track, multiple=False):
    """------------------------------------------------------------------------
    This function parses all types of pitch inputs that are fed into Chordia
    and Bozkurt functions. It can parse inputs of the following form:
    * A (list of) pitch track list(s)
    * A (list of) filename(s) for pitch track(s)
    * A (list of) PitchDistribution object(s)

    This is where we do the type checking to provide the correct input format
    to higher order functions. It returns to things, the cleaned input pitch
    track or pitch distribution.
    ------------------------------------------------------------------------"""
    # Multiple pitch tracks
    if multiple:
        return [parse_pitch_track(track, multiple=False)
                for track in pitch_track]

    # Single pitch track
    else:
        # Path to the pitch track
        if (type(pitch_track) == str) or (type(pitch_track) == unicode):
            if os.path.exists(pitch_track):
                result = np.loadtxt(pitch_track)
                # Strip the time track
                return result[:, 1] if result.ndim > 1 else result
            # Non-path string
            else:
                raise ValueError("Path doesn't exist: " + pitch_track)

        # Loaded pitch track passed
        elif (type(pitch_track) == np.ndarray) or (type(pitch_track) == list):
            return np.array(pitch_track)


def pd_zero_pad(pd, mode_pd):
    """------------------------------------------------------------------------
    This function is only used in tonic detection with pD. If necessary, it
    zero pads the distributions from both sides, to make them of the same
    length. The inputs are symmetric, i.e. the input distributions can be
    given in any order
    ---------------------------------------------------------------------------
    pD      : Input pD
    mode_pd : pD of the candidate mode
    ------------------------------------------------------------------------"""

    # The padding process requires the two bin lists to have an intersection.
    # This is ensured by the generate_pd function, since we enforce all pds to
    # include zero in their bins.

    # Finds the number of missing bins in the left and right sides of pD and
    # inserts that many zeros.
    diff_bins = set(mode_pd.bins) - set(pd.bins)
    num_left_missing = len([x for x in diff_bins if x < min(pd.bins)])
    num_right_missing = len([x for x in diff_bins if x > max(pd.bins)])
    pd.vals = np.concatenate((np.zeros(num_left_missing), pd.vals,
                              np.zeros(num_right_missing)))

    # Finds the number of missing bins in the left and right sides of mode_pd
    # and inserts that many zeros.
    diff_bins = set(pd.bins) - set(mode_pd.bins)
    num_left_missing = len([x for x in diff_bins if x < min(mode_pd.bins)])
    num_right_missing = len([x for x in diff_bins if x > max(mode_pd.bins)])
    mode_pd.vals = np.concatenate((np.zeros(num_left_missing), mode_pd.vals,
                                   np.zeros(num_right_missing)))

    return pd, mode_pd


def slice_pitch_track(time_track, pitch_track, chunk_size, threshold=0.5,
                      overlap=0):
    """------------------------------------------------------------------------
    Slices a pitch track into equal chunks of desired length.
    ---------------------------------------------------------------------------
    time_track  : The timestamps of the pitch track. This is used to determine
                  the samples to cut the pitch track. 1-D list
    pitch_track : The pitch track's frequency entries. 1-D list
    chunk_size  : The sizes of the chunks.
    threshold   : This is the ratio of smallest acceptable chunk to chunk_size.
                  When a pitch track is sliced the remaining tail at its end is
                  returned if its longer than threshold*chunk_size. Else, it's
                  discarded. However if the entire track is shorter than this
                  it is still returned as it is, in order to be able to
                  represent that recording.
    overlap     : If it's zero, the next chunk starts from the end of the
                  previous chunk, else it starts from the
                  (chunk_size*threshold)th sample of the previous chunk.
    ---------------------------------------------------------------------------
    chunks      : List of the pitch track chunks
    ------------------------------------------------------------------------"""
    chunks = []
    last = 0

    # Main slicing loop
    for k in np.arange(1, (int(max(time_track) / chunk_size) + 1)):
        cur = 1 + max(np.where(time_track < chunk_size * k)[0])
        chunks.append(pitch_track[last:(cur - 1)])

        # This variable keep track of where the first sample of the
        # next iteration should start from.
        last = 1 + max(np.where(
            time_track < chunk_size * k * (1 - overlap))[0]) \
            if (overlap > 0) else cur

    # Checks if the remaining tail should be discarded or not.
    if max(time_track) - time_track[last] >= chunk_size * threshold:
        chunks.append(pitch_track[last:])

    # If the runtime of the entire track is below the threshold, keep it as is
    elif last == 0:
        chunks.append(pitch_track)
    return chunks
