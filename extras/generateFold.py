import json
import os
from extras import fileOperations
from sklearn import cross_validation

def load_fold(fold_name):
	with open(fold_name, 'r') as f:
		res = json.load(f)
	return res

def stratified_fold(data_dir, n_folds = 10, savefile = ''):
	modes = fileOperations.getModeNames(data_dir)
	[filepaths, basefolders, filenames] = fileOperations.getFileNamesInDir(data_dir, extension='.pitch')

	filemodes = [os.path.basename(b) for b in basefolders]
	mode_idx = [modes.index(m) for m in filemodes]

	# get the stratified folds
	skf = cross_validation.StratifiedKFold(mode_idx, n_folds=n_folds)

	folds = dict()
	for ff, fold in enumerate(skf):
		folds['fold' + str(ff)] = {'train_set':[], 'test_set':[]}
		for tr_idx in fold[0]:
			folds['fold' + str(ff)]['train_set'].append({'mbid':mbids[tr_idx], 'makam':filemodes[tr_idx]})
		for te_idx in fold[1]:
			folds['fold' + str(ff)]['test_set'].append({'mbid':mbids[te_idx], 'makam':filemodes[te_idx]})

	# save the folds to a file if specified
	if savefile:
		with open(savefile, 'w') as f:
			json.dump(fold, f, indent=2)

	return folds