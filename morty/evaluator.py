# -*- coding: utf-8 -*-
from converter import Converter


class Evaluator(object):
    """----------------------------------------------------------------
    This class is used for evaluating the validity of our estimations.
    We return a dictionary entry as our evaluation result. See the
    return statements in each function to see which attributes are
    being reported.
    ----------------------------------------------------------------"""

    def __init__(self, tonic_tolerance=20):
        self.tonic_tolerance = tonic_tolerance
        self.CENT_PER_OCTAVE = 1200

        # '+' symbol corresponds to quarter tone higher
        self.INTERVAL_SYMBOLS = [
            ('P1', 0, 25), ('P1+', 25, 75), ('m2', 75, 125), ('m2+', 125, 175),
            ('M2', 175, 225), ('M2+', 225, 275), ('m3', 275, 325),
            ('m3+', 325, 375), ('M3', 375, 425), ('M3+', 425, 475),
            ('P4', 475, 525), ('P4+', 525, 575), ('d5', 575, 625),
            ('d5+', 625, 675), ('P5', 675, 725), ('P5+', 725, 775),
            ('m6', 775, 825), ('m6+', 825, 875), ('M6', 875, 925),
            ('M6+', 925, 975), ('m7', 975, 1025), ('m7+', 1025, 1075),
            ('M7', 1075, 1125), ('M7+', 1125, 1175), ('P1', 1175, 1200)]

    @staticmethod
    def evaluate_mode(estimated, annotated, source=None):
        mode_bool = annotated == estimated
        return {'source': source, 'mode_eval': mode_bool,
                'annotated_mode': annotated, 'estimated_mode': estimated}

    def evaluate_tonic(self, estimated, annotated, source=None):
        est_cent = Converter.hz_to_cent(estimated, annotated)

        # octave wrapping
        cent_diff = est_cent % self.CENT_PER_OCTAVE

        # check if the tonic is found correct
        bool_tonic = (min([cent_diff, self.CENT_PER_OCTAVE - cent_diff]) <
                      self.tonic_tolerance)

        # convert the cent difference to symbolic interval (P5, m3 etc.)
        interval = None
        for i in self.INTERVAL_SYMBOLS:
            if i[1] <= cent_diff < i[2]:
                interval = i[0]
                break
            elif cent_diff == 1200:
                interval = 'P1'
                break

        # if they are in the same octave the the estimated and octave-wrapped
        # values should be the same (very close)
        same_octave = (est_cent - cent_diff < 0.001)

        return {'mbid': source, 'tonic_eval': bool_tonic,
                'same_octave': same_octave, 'cent_diff': cent_diff,
                'interval': interval, 'annotated_tonic': annotated,
                'estimated_tonic': estimated}

    def evaluate_joint(self, tonic_info, mode_info, source=None):
        tonic_eval = self.evaluate_tonic(tonic_info[0], tonic_info[1], source)
        mode_eval = self.evaluate_mode(mode_info[0], mode_info[1], source)

        # merge the two evaluations
        joint_eval = tonic_eval.copy()
        joint_eval['mode_eval'] = mode_eval['mode_eval']
        joint_eval['annotated_mode'] = mode_eval['annotated_mode']
        joint_eval['estimated_mode'] = mode_eval['estimated_mode']
        joint_eval['joint_eval'] = (joint_eval['tonic_eval'] and
                                    joint_eval['mode_eval'])

        return joint_eval
