# -*- coding: utf-8 -*-
from predominantmelodymakam.predominantmelodymakam import PredominantMelodyMakam
from fileoperations.fileoperations import get_filenames_in_dir
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
elif len(sys.argv) == 2:  # for parallelization
    idx = int(sys.argv[1])
else:
    raise ValueError('Only accepts zero or one argument')

print(idx)

extractor = PredominantMelodyMakam()
audiodir = './'  # audio folder and sub folders

# text file
audio_files = get_filenames_in_dir(audiodir, keyword="*.mp3")[0]
txtfiles = [os.path.join(os.path.dirname(f), os.path.basename(
    os.path.splitext(f)[0]) + '.pitch') for f in audio_files]

if idx:  # if index is given
    audio_files = [audio_files[idx]]
    txtfiles = [txtfiles[idx]]

for ii, mp3 in enumerate(audio_files):
    print("{0:d}: {1:s}".format(ii + 1, os.path.basename(mp3)))

    if os.path.isfile(txtfiles[ii]):  # already exists
        print("   > Already exist; skipped.")
    else:
        results = extractor.run(mp3)

        pitch_track = np.array(json.loads(results['pitch']))[:, [0, 1]]
        pitch_track = (np.around([i * math.pow(10, DECIMAL)
                                  for i in pitch_track[:, 1]]) / 100.0)
        pitch_track = pitch_track.tolist()
        with open(txtfiles[ii], 'w') as f:
            for i in pitch_track:
                f.write("{0:.2f}\n".format(i))
