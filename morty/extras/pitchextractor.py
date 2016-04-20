# -*- coding: utf-8 -*-
from predominantmelodymakam.predominantmelodymakam \
    import PredominantMelodyMakam
from fileoperations.fileoperations import get_filenames_in_dir
import os
import numpy as np
import math


class PitchExtractor(object):
    extractor = PredominantMelodyMakam()
    DECIMAL = 1

    @classmethod
    def extract(cls, audiodir, start_idx=0):
        """
        Extract the predominant melody of all the audio recordings in the
        input folder and its subfolders
        :param audiodir: the audio directory
        :param start_idx: the index to start predominant melody extraction
        from the list of found audio recordings. This parameter is useful,
        if the user plans to run multiple instances of the extractor at once
        """
        # text file
        audio_files = get_filenames_in_dir(audiodir, keyword="*.mp3")[0]
        pitch_files = [os.path.join(os.path.dirname(f), os.path.basename(
            os.path.splitext(f)[0]) + '.pitch') for f in audio_files]

        if start_idx:  # if index is given
            audio_files = audio_files[start_idx:]
            pitch_files = pitch_files[start_idx:]

        for ii, (af, pf) in enumerate(zip(audio_files, pitch_files)):
            print(' ')
            print("{0:d}: {1:s}".format(ii + 1, os.path.basename(af)))

            if os.path.isfile(pf):  # already exists
                print("   > Already exist; skipped.")
            else:
                results = cls.extractor.run(af)3

                pitch_track = np.array(results['pitch'])[:, [0, 1]]
                pitch_track = (np.around([i * math.pow(10, cls.DECIMAL)
                                          for i in pitch_track[:, 1]]) / 100.0)

                decimal_str = '%.' + str(cls.DECIMAL) + 'f'
                np.savetxt(pf, pitch_track, fmt=decimal_str)
