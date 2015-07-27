# -*- coding: utf-8 -*-
import json
import numpy as np
import ModeFunctions as mf

class Evaluater:

	def __init__(self, tonic_tolerance=22.5):
		self.tolerance = tonic_tolerance
		CENT_PER_OCTAVE = 1200
		INTERVALS_SYMBOL = [('P1', 1150, 50), ('m2', 50, 150), ('M2', 150, 250), ('m3', 250, 350), ('M3', 350, 450), ('P4', 450, 550), ('d5', 550, 650), ('P5', 650, 750), ('m6', 750, 850), ('M6', 850, 950), ('m7', 950, 1050), ('M7', 1050, 1150)]

	def mode_evaluate(self, estimated, annotation):
		mbid = estimated['mbid']
		est_mode = estimated['estimation'][0][0]
		bool_mode = (annotation['makam'] == est_mode)
		return {'mbid':mbid, 'estimated_mode':bool_mode}

	def tonic_evaluate(self, estimated, annotation):
		mbid = estimated['mbid']
		est_tonic = estimated['estimation'][0][1]
		tmp_diff = mf.hz_to_cent([est_tonic], annotation['tonic'])[0] 
		cent_diff = ((cent_diff + CENT_PER_OCTAVE/2) % (CENT_PER_OCTAVE)) - (CENT_PER_OCTAVE/2)
		same_octave = (tmp_dif == cent_diff)
		bool_tonic = (abs(cent_diff) < self.tolerance)
		
		# finds the corresponsing interval symbol for the cent difference
		cent_diff = cent_diff if cent_diff > 0 else (1200+cent_diff)
		for i in INTERVAL_SYMBOLS:
			if (i[1] <=  cent_diff < i[2]):
				interval = i[0]
				break
		
		return {'mbid':mbid, 'estimated_tonic':bool_tonic, 'same_octave':same_octave, 'cent_diff': cent_diff, 'interval':interval}

	def joint_evaluate(self, estimated, annotation):
		tonic_eval = self.tonic_evaluate(estimated, annotation)
		mode_eval = self.mode_evaluate(estimated, annotation)
		
		# the two results are merged and returned
		result = tonic_eval.copy()
		result.update(mode_eval)
		return result


