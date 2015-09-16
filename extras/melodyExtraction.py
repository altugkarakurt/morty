# -*- coding: utf-8 -*-
from compmusic.extractors.makam import pitch
from extras.fileOperations import getFileNamesInDir
import os
import json
import math
import numpy as np

def batch_extract_melody(audioDir):
	extractor = pitch.PitchExtractMakam()

	audioFiles = getFileNamesInDir(audioDir, extension=".wav")[0]
	txtfiles = [os.path.splitext(f)[0] + '.txt' for f in audioFiles]  # ooutput files

	for ii, audio in enumerate(audioFiles):
		print ' '
		print str(ii+1) + ": " + os.path.basename(audio)

		results = extractor.run(audio)
		melody = np.array(json.loads(results['pitch']))[:, 1]

		# text file; only write until 2 decimal places to save from space
		with open(txtfiles[ii], 'w') as f:
			for i in melody:
				f.write("%.2f\n" % i)