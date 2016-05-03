# -*- coding: utf-8 -*-
from scipy.spatial import distance as spdistance
import collections
import numpy as np


class KNN(object):
    @classmethod
    def generate_distance_matrix(cls, distrib, peak_idxs, training_distribs,
                                 distance_method='bhat'):
        """--------------------------------------------------------------------
        Iteratively calculates the distance of the input distribution from each
        (mode candidate, tonic candidate) pair. This is a generic function,
        that is independent of distribution type or any other parameter value.
        -----------------------------------------------------------------------
        distribs            : Input distribution that is to be estimated
        peak_idxs           : List of indices of distribution peaks
        training_distribs   : List of training distributions
        method              : The distance method to be used. The available
                              distances are listed in distance() function.
        --------------------------------------------------------------------"""
        result = np.zeros((len(peak_idxs), len(training_distribs)))

        # Iterates over the peaks, i.e. the tonic candidates
        for i, cur_peak_idx in enumerate(peak_idxs):
            trial = distrib.shift(cur_peak_idx)

            # Iterates over mode candidates
            for j, td in enumerate(training_distribs):
                assert trial.bin_unit == td.bin_unit, \
                    'The bin units of the compared distributions should match.'
                assert trial.distrib_type() == td.distrib_type(), \
                    'The features should be of the same type'

                if trial.distrib_type() == 'pd':
                    # compare in the overlapping region
                    min_td_bin = np.min(td.bins)
                    max_td_bin = np.max(td.bins)

                    min_trial_bin = np.min(trial.bins)
                    max_trial_bin = np.max(trial.bins)

                    overlap = [max([min_td_bin, min_trial_bin]),
                               min([max_td_bin, max_trial_bin])]

                    trial_bool = (overlap[0] <= trial.bins) * \
                                 (trial.bins <= overlap[1])
                    trial_vals = trial.vals[trial_bool]

                    td_bool = (overlap[0] <= td.bins) * \
                              (td.bins <= overlap[1])
                    td_vals = td.vals[td_bool]
                else:
                    trial_vals = trial.vals
                    td_vals = td.vals

                # Calls the distance function for each entry of the matrix
                result[i][j] = cls._distance(trial_vals, td_vals,
                                             method=distance_method)
        return np.array(result)

    @staticmethod
    def _distance(vals_1, vals_2, method='bhat'):
        """--------------------------------------------------------------------
         Calculates the distance between two 1-D lists of values. This
         function is called with pitch distribution values, while generating
         distance matrices. The function is symmetric, the two inpıt lists
         are interchangable.
         ----------------------------------------------------------------------
         vals_1, vals_2 : The input value lists.
         method         : The choice of distance method
         ----------------------------------------------------------------------
         manhattan    : Minkowski distance of 1st degree
         euclidean    : Minkowski distance of 2nd degree
         l3           : Minkowski distance of 3rd degree
         bhat         : Bhattacharyya distance
         intersection : Intersection
         corr         : Correlation
         -------------------------------------------------------------------"""
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

    @staticmethod
    def get_nearest_neighbors(sorted_pair, k_param):
        # parse mode/tonic pairs
        pairs = [pair for pair, dist in sorted_pair[:k_param]]

        # find the most common pairs
        counter = collections.Counter(pairs)
        most_commons = counter.most_common(k_param)
        max_cnt = most_commons[0][1]
        cand_pairs = [c[0] for c in most_commons if c[1] == max_cnt]

        return cand_pairs

    @staticmethod
    def select_nearest_neighbor(cand_pairs, sorted_pair):
        # in case there are multiple candidates get the pair sorted earlier
        for p in sorted_pair:
            if p[0] in cand_pairs:
                estimated_pair = p

                # pop the estimated pair from the sorte_pair list for ranking
                sorted_pair = [pp for pp in sorted_pair if pp[0] != p[0]]
                return estimated_pair, sorted_pair

        assert False, 'No pair selected, this should be impossible!'
