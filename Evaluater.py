# -*- coding: utf-8 -*-
import json
import numpy as np
import ModeFunctions as mf

class Evaluater:

	def __init__(self, tonic_tolerance=25):
		self.tolerance = tonic_tolerance
		self.CENT_PER_OCTAVE = 1200
		# '+' symbol corresponds to quarter tone higher
		self.INTERVAL_SYMBOLS = [('P1', 0, 25), ('P1+', 25, 75), ('m2', 75, 125), ('m2+', 125, 175), ('M2', 175, 225),
		                         ('M2+', 225, 275), ('m3', 275, 325), ('m3+', 325, 375), ('M3', 375, 425), 
		                         ('M3+', 425, 475), ('P4', 475, 525), ('P4+', 525, 575), ('d5', 575, 625), 
		                         ('d5+', 625, 675), ('P5', 675, 725), ('P5+', 725, 775), ('m6', 775, 825), 
		                         ('m6+', 825, 875), ('M6', 875, 925), ('M6+', 925, 975), ('m7', 975, 1025), 
		                         ('m7+', 1025, 1075), ('M7', 1075, 1125), ('M7+', 1125, 1175), ('P1', 1175, 1200)]

	def mode_evaluate(self, mbid, estimated, annotated):
		mode_bool = (annotated == estimated)
		return {'mbid':mbid, 'mode_eval':mode_bool, 'annotated_mode':annotated, 'estimated_mode':estimated}

	def tonic_evaluate(self, mbid, estimated, annotated):
		tmp_diff = mf.hz_to_cent([estimated], annotated)[0] 
		cent_diff = ((tmp_diff + self.CENT_PER_OCTAVE/2) % (self.CENT_PER_OCTAVE)) - (self.CENT_PER_OCTAVE/2)
		same_octave = (tmp_diff - cent_diff < 0.01)
		bool_tonic = (abs(cent_diff) < self.tolerance)

		# finds the corresponsing interval symbol for the cent difference
		dummy_diff = cent_diff if cent_diff > 0 else (1200+cent_diff)
		for i in self.INTERVAL_SYMBOLS:
			if (i[1] <=  dummy_diff < i[2]):
				interval = i[0]
				break
			elif (dummy_diff == 1200):
				interval = 'P1'
				break
		
		return {'mbid':mbid, 'tonic_eval':bool_tonic, 'same_octave':same_octave, 'cent_diff': cent_diff, 'interval':interval, 'annotated_tonic':annotated, 'estimated_tonic':estimated}

	def joint_evaluate(self, mbid, tonic_info, mode_info):
		tonic_eval = self.tonic_evaluate(mbid, tonic_info[0], tonic_info[1])
		mode_eval = self.mode_evaluate(mbid, mode_info[0], mode_info[1])
		
		#merge the two evluations
		joint_eval = tonic_eval.copy()
		joint_eval['mode_eval'] = mode_eval['mode_eval']
		joint_eval['annotated_mode'] = mode_eval['annotated_mode']
		joint_eval['estimated_mode'] = mode_eval['estimated_mode']
		joint_eval['joint_eval'] = (joint_eval['tonic_eval'] and joint_eval['mode_eval'])

		return joint_eval


