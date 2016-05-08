import json
import os
from fileoperations.fileoperations import get_filenames_in_dir
from sklearn import cross_validation


class FoldGenerator(object):
    @classmethod
    def stratified_k_fold(cls, data_dir, annotation_in, n_folds=10,
                          random_state=None):
        """
        Generates stratified k folds from the audio_recordings in the
        data_dir. The stratification is applied according to the makam
        annotations
        :param data_dir: (str) data directory
        :param annotation_in: (str) json file or dictionary, which stores the
               annotations
               The loaded variable is a list of dictionaries, where each
               dictionary have the "mbid", "tonic" (frequency) and "makam"
               (name) keys, e.g.
               [
                 {
                   "mbid": "0db48ce4-f018-4d7d-b75e-66a64db72067",
                   "tonic": 151.1,
                   "makam": "Hicaz"
                 },
                 {
                   "mbid": "2c88acdf-685d-42c7-913d-1a9f2005587e",
                   "tonic": 292.5,
                   "makam": "Hicaz"
                 }
                 ...
               ]
        :param n_folds: (int) number of stratified folds requested
        :param random_state: (None, int or RandomState) pseudo-random number
               generator state used for shuffling. If None, use default numpy
               RNG for shuffling.
        :return: list of folds. each fold is organized as a dict with two keys
               "test" and "train". These keys store a list of dicts, where each
               dict has the "file", recording "MBID", (annotated) "tonic"
               and (annotated) "mode" keys, e.g:
               {'test': [
                   {'file': '0b45417b-acb4-4f8a-b180-5ad45be889af.pitch',
                    'mbid': u'0b45417b-acb4-4f8a-b180-5ad45be889af',
                    'mode': u'Saba',
                    'tonic': 328.3},
                   {'file': '3c25f0d8-a6df-4bde-87ef-e4af708b861d.pitch',
                    'mbid': u'3c25f0d8-a6df-4bde-87ef-e4af708b861d',
                    'mode': u'Hicaz',
                    'tonic': 150.0},
                    ...],
                'train': [
                   {...}]
        """
        modes = cls._get_mode_names(data_dir)
        [file_paths, base_folders, file_names] = get_filenames_in_dir(
            data_dir, keyword='*.pitch')

        try:  # json file
            annotations = json.load(open(annotation_in, 'r'))
        except TypeError:  # list of dict
            annotations = annotation_in

        file_modes, mbids, tonics = cls._parse_mbid_mode_tonic(
            annotations, file_names, base_folders)

        # get the stratified folds
        mode_idx = [modes.index(m) for m in file_modes]
        skf = cross_validation.StratifiedKFold(
            mode_idx, n_folds=n_folds, shuffle=True, random_state=random_state)

        folds = cls._organize_folds(skf, file_paths, mbids, file_modes, tonics)

        return folds

    @staticmethod
    def _organize_folds(k_folds, file_paths, mbids, file_modes, tonics):
        folds = []
        for ff, k_fold in enumerate(k_folds):
            train_ids, test_ids = k_fold

            # accumulate the training and testing data in the fold
            temp_fold = {
                # .train methods accept the inputs with the keys below. In
                # this format we can call the method as *.train(**training)
                'training': {'pitches': [], 'tonics': [], 'modes': [],
                             'sources': []},
                # .test methods accept a single data point. Organize it as a
                # list of dictionaries with the keys "pitch, tonic, mode,
                # source"
                'testing': []
            }
            for idx in train_ids:
                temp_fold['training']['pitches'].append(file_paths[idx])
                temp_fold['training']['tonics'].append(tonics[idx])
                temp_fold['training']['modes'].append(file_modes[idx])
                temp_fold['training']['sources'].append(mbids[idx])
            for idx in test_ids:
                temp_fold['testing'].append({
                    'pitch': file_paths[idx], 'tonic': tonics[idx],
                    'mode': file_modes[idx], 'source': mbids[idx]})

            folds.append(temp_fold)

        return folds

    @staticmethod
    def _parse_mbid_mode_tonic(annotations, file_names, base_folders):
        file_modes = [os.path.basename(b) for b in base_folders]
        mbids = [os.path.splitext(f)[0] for f in file_names]
        tonics = []
        for m in mbids:
            for a in annotations:
                if m in a['mbid']:  # a['mbid'] is a MusicBrainz link
                    tonics.append(a['tonic'])

        return file_modes, mbids, tonics

    @staticmethod
    def _get_mode_names(data_dir):
        # check if the folder exists
        if not os.path.isdir(data_dir):
            print("> Directory doesn't exist!")
            return []

        return [x[1] for x in os.walk(data_dir)][0]
