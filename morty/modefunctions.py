# -*- coding: utf-8 -*-
import numpy as np
import os
import copy
from scipy.spatial import distance as spdistance
from morty.pitchdistribution import PitchDistribution


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


def generate_distance_matrix(distrib, peak_idxs, mode_distribs,
                             method='bhat'):
    """------------------------------------------------------------------------
    Iteratively calculates the distance of the input distribution from each
    (mode candidate, tonic candidate) pair. This is a generic function, that is
    independent of distribution type or any other parameter value.
    ---------------------------------------------------------------------------
    distribs        : Input distribution that is to be estimated
    peak_idxs       : List of indices of distribution peaks
    mode_distribss  : List of candidate mode distributions
    method          : The distance method to be used. The available distances
                    are listed in distance() function.
    ------------------------------------------------------------------------"""

    result = np.zeros((len(peak_idxs), len(mode_distribs)))

    # Iterates over the peaks, i.e. the tonic candidates
    for i, cur_peak_idx in enumerate(peak_idxs):
        trial = distrib.shift(cur_peak_idx)

        # Iterates over mode candidates
        for j, cur_mode_dist in enumerate(mode_distribs):
            # Calls the distance function for each entry of the matrix
            result[i][j] = distance(trial.vals, cur_mode_dist.vals,
                                    method=method)
    return np.array(result)


def distance(vals_1, vals_2, method='bhat'):
    """------------------------------------------------------------------------
    Calculates the distance between two 1-D lists of values. This function is
    called with pitch distribution values, while generating distance matrices.
    The function is symmetric, the two inpıt lists are interchangable.
    ---------------------------------------------------------------------------
    vals_1, vals_2 : The input value lists.
    method         : The choice of distance method
    ---------------------------------------------------------------------------
    manhattan    : Minkowski distance of 1st degree
    euclidean    : Minkowski distance of 2nd degree
    l3           : Minkowski distance of 3rd degree
    bhat         : Bhattacharyya distance
    intersection : Intersection
    corr         : Correlation
    ------------------------------------------------------------------------"""
    if method == 'euclidean':
        return spdistance.euclidean(vals_1, vals_2)

    elif method == 'manhattan':
        return spdistance.minkowski(vals_1, vals_2, 1)

    elif method == 'l3':
        return spdistance.minkowski(vals_1, vals_2, 3)

    elif method == 'bhat':
        return -np.log(sum(np.sqrt(vals_1 * vals_2)))

    # Since correlation and intersection are actually similarity measures,
    # we take their inverse to be able to use them as distances. In other
    # words, max. similarity would give the min. inverse and we are always
    # looking for minimum distances.
    elif method == 'intersection':
        return len(vals_1) / (sum(np.minimum(vals_1, vals_2)))

    elif method == 'corr':
        return 1.0 - np.correlate(vals_1, vals_2)

    else:
        return 0


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


def tonic_estimate(distrib, peak_idxs, mode_distrib, distance_method="bhat"):
    """------------------------------------------------------------------------
    Given a mode (or candidate mode), compares the piece's distribution with
    each candidate tonic and returns the resultant distance vector to higher
    level functions. This is a wrapper function that handles the required
    preliminary tasks and calls generate_distance_matrix() accordingly.
    ---------------------------------------------------------------------------
    distrib         : Distribution of the input recording
    peak_idxs       : Indices of peaks (i.e. tonic candidates) of distribution
    mode_distib     : Distribution of the mode that distrib will be compared at
                      each iteration.
    distance_method : The choice of distance method. See the full list at
                      distance()
    ------------------------------------------------------------------------"""

    assert distrib.distrib_type() == mode_distrib.distrib_type(), \
        'Mismatch between the type of the input distribution and the trained '\
        'mode distribution.'

    # There are no preliminaries, simply generate the distance vector
    if distrib.distrib_type() == 'pcd':
        return np.array(generate_distance_matrix(
            distrib, peak_idxs, [mode_distrib], method=distance_method))[:, 0]
    elif distrib.distrib_type() == 'pd':
        # The PitchDistribution object is copied in order not to change its
        # internals before the following steps.
        temp = PitchDistribution(
            distrib.bins, distrib.vals, kernel_width=distrib.kernel_width,
            ref_freq=distrib.ref_freq)
        temp, mode_distrib = pd_zero_pad(temp, mode_distrib)

        # Fills both sides of distribution values with zeros, to make sure
        # that the shifts won't drop any non-zero values
        temp.vals = np.concatenate((np.zeros(abs(max(peak_idxs))), temp.vals,
                                    np.zeros(abs(min(peak_idxs)))))
        mode_distrib.vals = np.concatenate(
            (np.zeros(abs(max(peak_idxs))),
             mode_distrib.vals, np.zeros(abs(min(peak_idxs)))))

        return np.array(generate_distance_matrix(
            temp, peak_idxs, [mode_distrib], method=distance_method))[:, 0]


def mode_estimate(distrib, mode_distribs, distance_method='bhat'):
    """------------------------------------------------------------------------
    Compares the recording's distribution with each candidate mode with respect
    to the given tonic and returns the resultant distance vector to higher
    level functions. Here the input distribution is expected to be aligned
    according to the tonic and tonic  isn't explicitly used in this
    function. This is a wrapper function that handles the required
    preliminary tasks and calls generate_distance_matrix() accordingly.
    ---------------------------------------------------------------------------
    distrib         : Distribution of the input recording
    mode_distribs   : List of PitchDistribution objects. These are the model
                      pitch distributions of candidate modes.
    distance_method : The choice of distance method. See the full list at
                      distance()
    step_size       : The step-size of the pitch distribution. Unit is cents
    ------------------------------------------------------------------------"""

    assert all(distrib.distrib_type() == md.distrib_type()
               for md in mode_distribs), \
        'Mismatch between the type of the input distribution and the trained '\
        'mode distributions.'

    # There are no preliminaries, simply generate the distance vector.
    if distrib.distrib_type() == 'pcd':
        distance_vector = np.array(generate_distance_matrix(
            distrib, [0], mode_distribs, method=distance_method))[0]
    elif distrib.distrib_type() == 'pd':
        # For each trial, a new instance of PitchDistribution is created and
        # its attributes are copied from mode_distribs. For each trial, it
        # needs to be zero padded according to the current mode distribution
        # length. The entries of the vector is generated iteratively,
        # one-by-one.
        distance_vector = np.zeros(len(mode_distribs))
        for i, md in enumerate(mode_distribs):
            trial = copy.deepcopy(distrib)
            trial, mode_trial = pd_zero_pad(trial, md[i])
            distance_vector[i] = distance(trial.vals, mode_trial.vals,
                                          method=distance_method)
    else:
        raise ValueError('"distrib.type()" can either take the values "pd" or '
                         '"pcd".')

    return distance_vector


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
