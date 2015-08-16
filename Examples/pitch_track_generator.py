# -*- coding: utf-8 -*-
from compmusic.extractors.makam import pitch
import scipy
import json
import numpy as np

"""---------------------------------------------------------------------------------------
Example script for extracting predominant melody using CompMusic's pitch.py. Any other 
pitch extraction can also be used, as long as their output is of the same format.
---------------------------------------------------------------------------------------"""
extractor = pitch.PitchExtractMakam()
recordings = ['2-03_Ussak_Sazsemaisi']
save_dir = os.path.join('Pitch_Tracks', (recording + '.pitch'))
for recording in recordings:
	results = extractor.run(recording + '.mp3')
	pitch_track = np.array(json.loads(results['pitch']))[:, [0, 1]]
	pitch_track = (np.around([i*math.pow(10,DECIMAL) for i in pitch_track[:,1]]) / 100.0).tolist()

	with open(save_dir, 'w') as f:
		for i in pitch_track:
			f.write("%.2f\n" % i)

