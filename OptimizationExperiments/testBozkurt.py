# -*- coding: utf-8 -*-
import numpy as np
import sys
import json
import os
from os import path
sys.path.insert(0, './../')
import BozkurtEstimation as be

###Experiment Parameters-------------------------------------------------------------------------
rank = 10
fold_list = np.arange(1,11)
distance_list = ['manhattan', 'euclidan', 'l3', 'bhat', 'intersection', 'corr']

makam_list = ['Acemasiran', 'Acemkurdi', 'Beyati', 'Bestenigar', 'Hicaz', 
			  'Hicazkar', 'Huseyni', 'Huzzam', 'Karcigar', 'Kurdilihicazkar', 
			  'Mahur', 'Muhayyer', 'Neva', 'Nihavent', 'Rast', 'Saba', 
			  'Segah', 'Sultaniyegah', 'Suzinak', 'Ussak']

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!DATA FOLDER INIT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#data_folder = '../../../Makam_Dataset/Pitch_Tracks/'
#data_folder = '../../../test_datasets/turkish_makam_recognition_dataset/data/' sertan desktop local
#data_folder = os.path.join('..', '..', '..', experiments, 'turkish_makam_recognition_dataset', 'data') # hpc cluster

# folder structure
experiment_dir = './Experiments' # assumes it is already created

#chooses which training to use 
training_idx = int(sys.argv[1])
training_dir = os.path.join(experiment_dir, 'Training' + str(training_idx))

# get the training experient/fold parameters 
with open(os.path.join(training_dir, 'parameters.json'), 'r') as f:
	cur_params = json.load(f)
	f.close()

for distance in distance_list:
	cent_ss = cur_params['cent_ss']
	smooth_factor = cur_params['smooth_factor']
	distribution_type = cur_params['distribution_type']
	chunk_size = chunk_size_list['chunk_size']

	# instantiate makam estimator for training
	estimator = be.BozkurtEstimation(cent_ss=cent_ss, smooth_factor=smooth_factor, 
		                             chunk_size=chunk_size)

	# load annotations; the tonic values will be read from here
	with open('annotations.json', 'r') as f:
		annot = json.load(f)
		f.close()

	output = dict()
	for fold in fold_list:
		output['Fold' + str(fold)] = []
		fold_dir = os.path.join(experiment_dir, 'Fold' + str(fold))
		
		# load the current fold to get the training recordings
		with open((os.path.join('./Folds', 'fold_' + str(fold) + '.json')), 'r') as f:
			cur_fold = json.load(f)['test']
			f.close()
		# retrieve annotations of the training recordings
		for makam_name in makam_list:
			# divide the training data into makams
			makam_annot = [k for k in cur_fold if k['makam']==makam_name]
			pitch_track_dir = os.path.join(data_folder, makam_name)

			for i in makam_annot:
				for j in annot:
					# append the tonic of the recordıng from the relevant annotation
					if(i['mbid'] == j['mbid']):
						i['tonic'] = j['tonic'] 
						break

			#actual estimation
			for recording in makam_annot:
				pitch_track = mf.load_track(txt_name=(recording['mbid'] + '.pitch'), 
					                        txt_dir=pitch_track_dir)[:,1]
				#estimate makam
				cur_out = estimator.estimate(pitch_track, mode_names=makam_list, 
					         est_tonic=False, est_mode=True, rank=rank, 
					         distance_method=distance, metric=distribution_type, 
					         ref_freq = recording["tonic"])
				output[('Fold' + fold)].append({recording['mbid']:cur_out})
	with open(os.path.join(training_dir, distance), 'w') as f:
		json.dump(output, f, indent=2)
		f.close()
	print '   Finished! ' + 'training: ' + training_idx

