# -*- coding: utf-8 -*-
import numpy as np
import os
from scipy.spatial import distance as spdistance
from scipy.integrate import simps
from scipy.stats import norm

from modetonicestimation.PitchDistribution import PitchDistribution


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


def generate_pd(cent_track, ref_freq=440, smooth_factor=7.5, step_size=7.5):
    """------------------------------------------------------------------------
    Given the pitch track in the unit of cents, generates the Pitch
    Distribution of it. the pitch track from a text file. 0th column is the
    time-stamps and
    1st column is the corresponding frequency values.
    ---------------------------------------------------------------------------
    cent_track:     1-D array of frequency values in cents.
    ref_freq:       Reference frequency used while converting Hz values to
                    cents.
                    This number isn't used in the computations, but is to be
                    recorded in the PitchDistribution object.
    smooth_factor:  The standard deviation of the gaussian kernel, used in
                    Kernel Density Estimation. If 0, a histogram is given
    step_size:      The step size of the Pitch Distribution bins.
    source:         The source information (i.e. recording name/id) to be
                    stored in PitchDistribution object.
    segment:        Stores which part of the recording, the distribution
                    belongs to. It stores the endpoints in seconds, such as
                    [0,60].
                    This is only useful for Chordia Estimation.
    overlap:        The ratio of overlap (hop size / chunk size) to be stored.
                    This is only useful for Chordia Estimation.
    ------------------------------------------------------------------------"""

    # Some extra interval is added to the beginning and end since the
    # superposed Gaussian for smooth_factor would introduce some tails in the
    # ends. These vanish after 3 sigmas(=smooth_factor).

    # The limits are also quantized to be a multiple of chosen step-size
    # smooth_factor = standard deviation of the gaussian kernel

    # TODO: filter out the NaN, -infinity and +infinity from the pitch track

    # Finds the endpoints of the histogram edges. Histogram bins will be
    # generated as the midpoints of these edges.
    min_edge = min(cent_track) - (step_size / 2.0)
    max_edge = max(cent_track) + (step_size / 2.0)
    pd_edges = np.concatenate(
        [np.arange(-step_size / 2.0, min_edge, -step_size)[::-1],
         np.arange(step_size / 2.0, max_edge, step_size)])

    # An exceptional case is when min_bin and max_bin are both positive
    # In this case, pd_edges would be in the range of [step_size/2, max_bin].
    # If so, a -step_size is inserted to the head, to make sure 0 would be
    # in pd_bins. The same procedure is repeated for the case when both
    # are negative. Then, step_size is inserted to the tail.
    pd_edges = pd_edges if -step_size / 2.0 in pd_edges else np.insert(
        pd_edges, 0, -step_size / 2.0)
    pd_edges = pd_edges if step_size / 2.0 in pd_edges else np.append(
        pd_edges, step_size / 2.0)

    # Generates the histogram and bins (i.e. the midpoints of edges)
    pd_vals, pd_edges = np.histogram(cent_track, bins=pd_edges, density=True)
    pd_bins = np.convolve(pd_edges, [0.5, 0.5])[1:-1]

    if smooth_factor > 0:  # kernel density estimation (approximated)
        # smooth the histogram
        normal_dist = norm(loc=0, scale=smooth_factor)
        xn = np.concatenate(
            [np.arange(0, - 5 * smooth_factor, -step_size)[::-1],
             np.arange(step_size, 5 * smooth_factor, step_size)])
        sampled_norm = normal_dist.pdf(xn)
        if len(sampled_norm) <= 1:
            raise ValueError("the smoothing factor is too small compared to "
                             "the step size, such that the convolution "
                             "kernel returns a single point gaussian. Either "
                             "increase the value to at least (step size/3) "
                             "or assign smooth factor to 0, for no smoothing.")

        extra_num_bins = len(sampled_norm) / 2  # convolution generates tails
        pd_vals = np.convolve(pd_vals,
                              sampled_norm)[extra_num_bins:-extra_num_bins]

        # normalize the area under the curve
        area = simps(pd_vals, dx=step_size)
        pd_vals = pd_vals / area

    # Sanity check. If the histogram bins and vals lengths are different, we
    # are in trouble. This is an important assumption of higher level functions
    if len(pd_bins) != len(pd_vals):
        raise ValueError('Lengths of bins and Vals are different')

    # Initializes the PitchDistribution object and returns it.
    return PitchDistribution(pd_bins, pd_vals, kernel_width=smooth_factor,
                             ref_freq=ref_freq)


def generate_pcd(pd):
    """------------------------------------------------------------------------
    Given the pitch distribution of a recording, generates its pitch class
    distribution, by octave wrapping.
    ---------------------------------------------------------------------------
    pD: PitchDistribution object. Its attributes include everything we need
    ------------------------------------------------------------------------"""

    # Initializations
    pcd_bins = np.arange(0, 1200, pd.step_size)
    pcd_vals = np.zeros(len(pcd_bins))

    # Octave wrapping
    for k in range(len(pd.bins)):
        idx = int((pd.bins[k] % 1200) / pd.step_size)
        idx = idx if idx != 160 else 0
        pcd_vals[idx] += pd.vals[k]

    # Initializes the PitchDistribution object and returns it.
    return PitchDistribution(pcd_bins, pcd_vals, kernel_width=pd.kernel_width,
                             ref_freq=pd.ref_freq)


def hz_to_cent(hz_track, ref_freq):
    """------------------------------------------------------------------------
    Converts an array of Hertz values into cents.
    ---------------------------------------------------------------------------
    hz_track : The 1-D array of Hertz values
    ref_freq    : Reference frequency for cent conversion
    ------------------------------------------------------------------------"""
    hz_track = np.array(hz_track)

    # The 0 Hz values are removed, not only because they are meaningless,
    # but also logarithm of 0 is problematic.
    return np.log2(hz_track[hz_track > 0] / ref_freq) * 1200.0


def cent_to_hz(cent_track, ref_freq):
    """------------------------------------------------------------------------
    Converts an array of cent values into Hertz.
    ---------------------------------------------------------------------------
    cent_track  : The 1-D array of cent values
    ref_freq    : Reference frequency for cent conversion
    ------------------------------------------------------------------------"""
    cent_track = np.array(cent_track)

    return 2 ** (cent_track / 1200.0) * ref_freq


def generate_distance_matrix(dist, peak_idxs, mode_dists, method='euclidean'):
    """------------------------------------------------------------------------
    Iteratively calculates the distance of the input distribution from each
    (mode candidate, tonic candidate) pair. This is a generic function, that is
    independent of distribution type or any other parameter value.
    ---------------------------------------------------------------------------
    dist       : Input distribution that is to be estimated
    peak_idxs  : List of indices of dist's peaks
    mode_dists : List of candidate mode distributions
    method     : The distance method to be used. The available distances are
                 listed in distance() function.
    ------------------------------------------------------------------------"""

    result = np.zeros((len(peak_idxs), len(mode_dists)))

    # Iterates over the peaks, i.e. the tonic candidates
    for i, cur_peak_idx in enumerate(peak_idxs):
        trial = dist.shift(cur_peak_idx)

        # Iterates over mode candidates
        for j, cur_mode_dist in enumerate(mode_dists):
            # Calls the distance function for each entry of the matrix
            result[i][j] = distance(trial.vals, cur_mode_dist.vals,
                                    method=method)
    return np.array(result)


def distance(vals_1, vals_2, method='euclidean'):
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


def tonic_estimate(dist, peak_idxs, mode_dist, distance_method="euclidean",
                   metric='pcd'):
    """------------------------------------------------------------------------
    Given a mode (or candidate mode), compares the piece's distribution with
    each candidate tonic and returns the resultant distance vector to higher
    level functions. This is a wrapper function that handles the required
    preliminary tasks and calls generate_distance_matrix() accordingly.
    ---------------------------------------------------------------------------
    dist            : Distribution of the input recording
    peak_idxs       : Indices of peaks (i.e. tonic candidates) of dist
    mode_dist       : Distribution of the mode that dist will be compared at
                      each iteration.
    distance_method : The choice of distance method. See the full list at
                      distance()
    metric          : Whether PCD or PD is used
    step_size         : The step-size of the pitch distribution. Unit is cents
    ------------------------------------------------------------------------"""

    # TODO: step_size and pD/pcd information can be retrieved from the dist
    # object
    # try and test that

    # There are no preliminaries, simply generate the distance vector
    if metric == 'pcd':
        return np.array(generate_distance_matrix(dist, peak_idxs, [mode_dist],
                                                 method=distance_method))[:, 0]
    elif metric == 'pd':
        # The PitchDistribution object is copied in order not to change its
        # internals before the following steps.
        temp = PitchDistribution(
            dist.bins, dist.vals, kernel_width=dist.kernel_width,
            ref_freq=dist.ref_freq)
        temp, mode_dist = pd_zero_pad(temp, mode_dist)

        # Fills both sides of distribution values with zeros, to make sure
        # that the shifts won't drop any non-zero values
        temp.vals = np.concatenate((np.zeros(abs(max(peak_idxs))), temp.vals,
                                    np.zeros(abs(min(peak_idxs)))))
        mode_dist.vals = np.concatenate(
            (np.zeros(abs(max(peak_idxs))),
             mode_dist.vals, np.zeros(abs(min(peak_idxs)))))

        return np.array(generate_distance_matrix(temp, peak_idxs, [mode_dist],
                                                 method=distance_method))[:, 0]


def mode_estimate(dist, mode_dists, distance_method='euclidean', metric='pcd'):
    """------------------------------------------------------------------------
    Compares the recording's distribution with each candidate mode with respect
    to the given tonic and returns the resultant distance vector to higher
    level functions. Here the input distribution is expected to be aligned
    according to the tonic and tonic  isn't explicitly used in this
    function. This is a wrapper function that handles the required
    preliminary tasks and calls generate_distance_matrix() accordingly.
    ---------------------------------------------------------------------------
    dist            : Distribution of the input recording
    mode_dists      : List of PitchDistribution objects. These are the model
                      pitch distributions of candidate modes.
    distance_method : The choice of distance method. See the full list at
                      distance()
    metric          : Whether PCD or PD is used
    step_size         : The step-size of the pitch distribution. Unit is cents
    ------------------------------------------------------------------------"""

    # TODO: step_size and pD/pcd information can be retrieved from the dist
    # object try and test that

    # There are no preliminaries, simply generate the distance vector.
    if metric == 'pcd':
        distance_vector = np.array(generate_distance_matrix(
            dist, [0], mode_dists, method=distance_method))[0]

    elif metric == 'pD':
        distance_vector = np.zeros(len(mode_dists))

        # For each trial, a new instance of PitchDistribution is created and
        # its attributes are copied from dist. For each trial, it needs to
        # be zero padded according to the current mode distribution length.
        # The entries of the vector is generated iteratively, one-by-one.
        for i in range(len(mode_dists)):
            trial = PitchDistribution(
                dist.bins, dist.vals, kernel_width=dist.kernel_width,
                ref_freq=dist.ref_freq)
            trial, mode_trial = pd_zero_pad(trial, mode_dists[i])
            distance_vector[i] = distance(trial, mode_trial,
                                          method=distance_method)
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
