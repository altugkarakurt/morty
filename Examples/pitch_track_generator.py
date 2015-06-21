# -*- coding: utf-8 -*-
from compmusic.extractors.makam import pitch
import scipy
import json
import numpy as np

"""---------------------------------------------------------------------------------------
Example script for extracting predominant melody using CompMusic's pitch.py. Any other 
pitch extraction can also be used, as long as their output are of the same format.
---------------------------------------------------------------------------------------"""
extractor = pitch.PitchExtractMakam()
test_songs = ['semahat', 'murat_derya', 'gec_kalma']

for song in test_songs:
	results = extractor.run(song + '.mp3')
	pitch_track = np.array(json.loads(results['pitch']))[:, [0, 1]]
	with open(('./Pitch Tracks/' + song + '.txt'), 'w') as f:
		np.savetxt(f, pitch_track)
