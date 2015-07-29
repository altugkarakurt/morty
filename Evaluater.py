# -*- coding: utf-8 -*-
import json
import numpy as np
import ModeFunctions as mf

class Evaluater:

	def __init__(self, tonic_tolerance=22.5):
		self.tolerance = tonic_tolerance
		self.CENT_PER_OCTAVE = 1200
		self.INTERVAL_SYMBOLS = [('P1', 0, 50), ('m2', 50, 150), ('M2', 150, 250), ('m3', 250, 350), ('M3', 350, 450), ('P4', 450, 550), ('d5', 550, 650), ('P5', 650, 750), ('m6', 750, 850), ('M6', 850, 950), ('m7', 950, 1050), ('M7', 1050, 1150), ('P1', 1150, 1200)]

	def mode_evaluate(self, mbid, estimated, annotated):
		return {'mbid':mbid, 'mode_eval':(annotated == estimated)}

	def tonic_evaluate(self, mbid, estimated, annotated):
		tmp_diff = mf.hz_to_cent([estimated], annotated)[0] 
		cent_diff = ((tmp_diff + self.CENT_PER_OCTAVE/2) % (self.CENT_PER_OCTAVE)) - (self.CENT_PER_OCTAVE/2)
		same_octave = (tmp_diff == cent_diff)
		bool_tonic = (abs(cent_diff) < self.tolerance)
		
		# finds the corresponsing interval symbol for the cent difference
		cent_diff = cent_diff if cent_diff > 0 else (1200+cent_diff)
		for i in self.INTERVAL_SYMBOLS:
			if (i[1] <=  cent_diff < i[2]):
				interval = i[0]
				break
			elif (cent_diff == 1200):
				interval = 'P1'
				break
		
		return {'mbid':mbid, 'tonic_eval':bool_tonic, 'same_octave':same_octave, 'cent_diff': cent_diff, 'interval':interval}

	def joint_evaluate(self, mbid, tonic_info, mode_info):
		tonic_eval = self.tonic_evaluate(mbid, tonic_info[0], tonic_info[1])
		mode_eval = self.mode_evaluate(mbid, mode_info[0], mode_info[1])
		
		#merge the two evluations
		joint_eval = tonic_eval.copy()
		joint_eval['mode_eval'] = mode_eval['mode_eval']
		joint_eval['joint_eval'] = (joint_eval['tonic_eval'] and joint_eval['mode_eval'])

		return joint_eval


