# -*- coding: utf-8 -*-
from predominantmelodymakam.predominantmelodymakam \
    import PredominantMelodyMakam
from pitchfilter.pitchfilter import PitchFilter
from fileoperations.fileoperations import get_filenames_in_dir
import os
import numpy as np


class Pitch(object):
    extractor = PredominantMelodyMakam(filter_pitch=False)  # call the
    # Python implementation of pitch_filter explicitly
    filter = PitchFilter()
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
                # extract and filter
                results = cls.extractor.run(af)
                pitch_track = cls.filter.run(results['pitch'])

                # save compact
                pitch_track = np.array(pitch_track)[:, 1]
                decimal_str = '%.' + str(cls.DECIMAL) + 'f'

                np.savetxt(pf, pitch_track, fmt=decimal_str)

    @staticmethod
    def slice(time_track, pitch_track, chunk_size, threshold=0.5,
              overlap=0):
        """--------------------------------------------------------------------
        Slices a pitch track into equal chunks of desired length.
        -----------------------------------------------------------------------
        time_track  : The timestamps of the pitch track. This is used to
                      determine the samples to cut the pitch track. 1-D list
        pitch_track : The pitch track's frequency entries. 1-D list
        chunk_size  : The sizes of the chunks.
        threshold   : This is the ratio of smallest acceptable chunk to
                      chunk_size. When a pitch track is sliced the remaining
                      tail at its end is returned if its longer than
                      (threshold * chunk_size). Else, it's discarded.
                      However if the entire track is shorter than this it is
                      still returned as it is, in order to be able to
                      represent that recording.
        overlap     : If it's zero, the next chunk starts from the end of the
                      previous chunk, else it starts from the
                      (chunk_size*threshold)th sample of the previous chunk.
        -----------------------------------------------------------------------
        chunks      : List of the pitch track chunks
        --------------------------------------------------------------------"""
        chunks = []
        last = 0

        # Main slicing loop
        for k in np.arange(1, (int(max(time_track) / chunk_size) + 1)):
            cur = 1 + max(np.where(time_track < chunk_size * k)[0])
            chunks.append(pitch_track[last:(cur - 1)])

            # This variable keep track of where the first sample of the
            # next iteration should start from.
            last = 1 + max(np.where(
                time_track < chunk_size * k * (1 - overlap))[0]) \
                if (overlap > 0) else cur

        # Checks if the remaining tail should be discarded or not.
        if max(time_track) - time_track[last] >= chunk_size * threshold:
            chunks.append(pitch_track[last:])

        # If the runtime of the entire track is below the threshold, keep it
        elif last == 0:
            chunks.append(pitch_track)
        return chunks
