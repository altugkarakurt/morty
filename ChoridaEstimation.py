# -*- coding: utf-8 -*-
import numpy as np

import ModeFunctions as mf
import PitchDistribution as p_d

class ChordiaEstimation:

	def __init__(self, cent_ss=7.5, smooth_factor=7.5, chunk_size=60):
		self.cent_ss = cent_ss
		self.smooth_factor = smooth_factor
		self.chunk_size = chunk_size

	