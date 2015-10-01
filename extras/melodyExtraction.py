# -*- coding: utf-8 -*-

from compmusic.extractors.makam import pitch
from FileOperations import getFileNamesInDir
import os 
import json
import numpy as np
import sys
import math

# Number of decimal points to round the pitch track to.
DECIMAL = 2

# get the input index
if len(sys.argv) == 1:
	idx = []
elif len(sys.argv) == 2: # for paralelization
	idx = int(sys.argv[1])
else:
	raise ValueError('Only accepts zero or one argument')

print idx

extractor = pitch.PitchExtractMakam()

audioDir = './' # audio folder and subfolders

audioFiles = getFileNamesInDir(audioDir, audio_ext=".mp3")[0]
txtFiles = [os.path.join(os.path.dirname(f), os.path.basename(os.path.splitext(f)[0])+'.pitch') for f in audioFiles] # text file; for sonic visualizer

if idx: # if index is given
	audioFiles = [audioFiles[idx]]
	txtFiles = [txtFiles[idx]]

for ii, mp3 in enumerate(audioFiles):
	print ' '
	print str(ii+1) + ": " + os.path.basename(mp3)

	if os.path.isfile(txtFiles[ii]): # already exists
		print "   > Already exist; skipped."
	else:
		results = extractor.run(mp3)
		pitch = np.array(json.loads(results['pitch']))[:, [0, 1]]
		pitch_track = np.array(json.loads(results['pitch']))[:, [0, 1]]
		pitch_track = (np.around([i*math.pow(10,DECIMAL) \
			          for i in pitch_track[:,1]]) / 100.0).tolist()
		with open(txtFiles[ii], 'w') as f:
			for i in pitch_track:
				f.write("%.2f\n" % i)
