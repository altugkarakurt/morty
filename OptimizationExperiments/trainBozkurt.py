# -*- coding: utf-8 -*-
import numpy as np
import sys
import json
import os
from os import path
sys.path.insert(0, './../')
import BozkurtEstimation as be

###Experiment Parameters-------------------------------------------------------------------------
fold_list = np.arange(1,11)
cent_ss_list = [7.5, 15, 25, 50, 100]
smooth_factor_list = [0, 2.5, 7.5, 15, 20]
metric_list = ['pcd', 'pd']
chunk_size_list = [30, 60, 90, 120, 0]
makam_list = ['Acemasiran', 'Acemkurdi', 'Beyati', 'Bestenigar', 'Hicaz', 
			  'Hicazkar', 'Huseyni', 'Huzzam', 'Karcigar', 'Kurdilihicazkar', 
			  'Mahur', 'Muhayyer', 'Neva', 'Nihavent', 'Rast', 'Saba', 
			  'Segah', 'Sultaniyegah', 'Suzinak', 'Ussak']

data_folder = '../../../Makam_Dataset/Pitch_Tracks/'
#data_folder = '../../../test_datasets/turkish_makam_recognition_dataset/data/' sertan desktop local
#data_folder = os.path.joın('..', '..', '..', experiments, 'turkish_makam_recognition_dataset', 'data') # hpc cluster

# get the training experient/fold parameters 
idx = np.unravel_index(int(sys.argv[1]), (len(fold_list), len(cent_ss_list), len(smooth_factor_list), len(metric_list), len(chunk_size_list)))
fold = fold_list[idx[0]]
cent_ss = cent_ss_list[idx[1]]
smooth_factor = smooth_factor_list[idx[2]]
metric = metric_list[idx[3]]
chunk_size = chunk_size_list[idx[4]]

# instantiate makam estimator for training
estimator = be.BozkurtEstimation(cent_ss=cent_ss, smooth_factor=smooth_factor, chunk_size=chunk_size)

# experiment info
experiment_info = {'cent_ss': cent_ss, 'smooth_factor':smooth_factor, 'metric':metric, 'chunk_size':chunk_size, 'method':'bozkurt'}

# folder structure
experiment_master_dir = './Experiments' # assumes it is already created

experiment_dir = os.path.join(experiment_master_dir, 'Experiment' + sys.argv[1])
if not os.path.exists(experiment_dir):
	os.makedirs(experiment_dir)

# save the experiment info
with open(os.path.join(experiment_dir, 'parameters.json'), 'w') as f:
	json.dump(experiment_info, f, indent=2)
	f.close()

# create the training folder
fold_dir = os.path.join(experiment_dir, 'Fold' + str(fold))
if not os.path.exists(fold_dir):
	os.makedirs(fold_dir)

# check if the training has already been done by comparing the names of the json
# files in the fold directory. If finished, skip training
training_filenames = next(os.walk(fold_dir))[2]
makam_names = [os.path.splitext(os.path.split(f)[1])[0] for f in training_filenames]
if (set(makam_list) - set(makam_names) == set()):
	sys.exit()

# load annotations; the tonic values will be read from here
with open('annotations.json', 'r') as f:
	annot = json.load(f)
	f.close()

# load the fold to get the training recordings
with open((os.path.join('./Folds', 'fold_' + str(fold) + '.json')), 'r') as f:
	cur_fold = json.load(f)['test']
	f.close()

print 'Vamos ! ' + 'training: ' + str(sys.argv[1])

# retrieve annotations of the training recordings
for makam_name in makam_list:
	# dıvıde the traınıng data into makams
	makam_annot = [k for k in cur_fold if k['makam']==makam_name]
	for i in makam_annot:
		for j in annot:
			# append the tonic of the recordıng from the relevant annotation
			if(i['mbid'] == j['mbid']):
				i['tonic'] = j['tonic'] 
				break

	# get pitch tracks and tonic frequencies
	pitch_track_dir = os.path.join(data_folder, makam_name)
	pitch_track_list = [(ma['mbid'] + '.pitch') for ma in makam_annot]
	tonic_list = [ma['tonic'] for ma in makam_annot]

	# train
	estimator.train(makam_name, pitch_track_list, tonic_list, metric=metric, 
		pt_dir=pitch_track_dir, save_dir=fold_dir)

print '   Finished! ' + 'training: ' + str(sys.argv[1])
