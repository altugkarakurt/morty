# -*- coding: utf-8 -*-
import numpy as np
import sys
import json
import time
import os
from os import path
sys.path.insert(0, './../')
import ChordiaEstimation as che
import ModeFunctions as mf


###Experiment Parameters-------------------------------------------------------------------------
threshold = 0.5
distance_list = ['intersection', 'manhattan', 'bhat']
makam_list = ['Acemasiran', 'Acemkurdi', 'Beyati', 'Bestenigar', 'Hicaz', 
			  'Hicazkar', 'Huseyni', 'Huzzam', 'Karcigar', 'Kurdilihicazkar', 
			  'Mahur', 'Muhayyer', 'Neva', 'Nihavent', 'Rast', 'Saba', 
			  'Segah', 'Sultaniyegah', 'Suzinak', 'Ussak']
k_list = [1,3,5,10]
training_list = np.arange(1,49)
fold_list = np.arange(1,11)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!DATA FOLDER INIT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#data_folder = '../../../Makam_Dataset/Pitch_Tracks/'
#data_folder = '../../../test_datasets/turkish_makam_recognition_dataset/data/' #sertan desktop local
data_folder = '../../../experiments/turkish_makam_recognition_dataset/data/' # hpc cluster

x = int(sys.argv[1])

# folder structure
experiment_dir = './ChordiaExperiments' # assumes it is already created

#chooses which training to use
idx = np.unravel_index(int(x-1), (len(k_list), len(distance_list), len(training_list), len(fold_list)))

k_param = k_list[idx[0]]
distance = distance_list[idx[1]]
training_idx = training_list[idx[2]]
fold = fold_list[idx[3]]

training_dir = os.path.join(experiment_dir, 'Training' + str(training_idx))
tonicPath = os.path.join(training_dir, 'Tonic')
if not os.path.exists(tonicPath):
	os.makedirs(tonicPath)

distancePath = os.path.join(tonicPath, (distance + '_k' + str(k_param)))
if not os.path.exists(distancePath):
	os.makedirs(distancePath)

# get the training experient/fold parameters 
with open(os.path.join(training_dir, 'parameters.json'), 'r') as f:
	cur_params = json.load(f)
	f.close()

json_dir = os.path.join(tonicPath, (distance + '_k' + str(k_param) + '.json'))
if os.path.isfile(json_dir):
	print 'Exists!'
	sys.exit()

if os.path.isfile(os.path.join(distancePath, (str(fold) + '.json'))):
	print 'Exists! Yay! ' + os.path.join(distancePath, (str(fold) + '.json'))
	sys.exit()

done_dists = next(os.walk(tonicPath))[2]
done_dists = [d[:-5] for d in done_dists]
if (distance in done_dists):
	print 'Already done ' + distance
	sys.exit()

print 'Computing ' + distance
cent_ss = cur_params['cent_ss']
smooth_factor = cur_params['smooth_factor']
distribution_type = cur_params['distribution_type']
chunk_size = cur_params['chunk_size']
overlap = cur_params['overlap']

# instantiate makam estimator for training
estimator = che.ChordiaEstimation(cent_ss=cent_ss, smooth_factor=smooth_factor, chunk_size=chunk_size, threshold=threshold, overlap=overlap)

# load annotations; the tonic values will be read from here
with open('annotations.json', 'r') as f:
	annot = json.load(f)
	f.close()

output = dict()
for fold in fold_list:
	output['Fold' + str(fold)] = []
	fold_dir = os.path.join(training_dir, 'Fold' + str(fold))
	
	# load the current fold to get the test recordings
	with open((os.path.join('./Folds', 'fold_' + str(fold) + '.json')), 'r') as f:
		cur_fold = json.load(f)['test']
		f.close()

	# retrieve annotations of the training recordings
	for makam_name in makam_list:

		# just for checking the uniqueness of test recordings
		with open(os.path.join(fold_dir, makam_name + '.json')) as f:
			makam_recordings = json.load(f)[0]['source']
			f.close()

		# divide the training data into makams
		makam_annot = [k for k in cur_fold if k['makam']==makam_name]
		pitch_track_dir = os.path.join(data_folder, makam_name)

		# load the annotations for testing data; it will be only used for 
		# makam recognition (with annotated tonic)
		for i in makam_annot:
			for j in annot:
				# append the tonic of the recordıng from the relevant annotation
				if(i['mbid'] == j['mbid']):
					i['tonic'] = j['tonic'] 
					break

		#actual estimation
		for recording in makam_annot:
			
			#check if test recording was use in training
			if (recording['mbid'] + '.pitch' in makam_recordings):
				raise ValueError(('Unique-check Failure. ' + recording['mbid']))

			pitch_track = mf.load_track(txt_name=(recording['mbid'] + '.pitch'), 
				                        txt_dir=pitch_track_dir)
			init_time = time.time()
			cur_out = estimator.estimate(pitch_track[:,1], pitch_track[:,0], 
						mode_names=makam_list, est_tonic=True, est_mode=False, k_param=k_param,
						distance_method=distance, metric=distribution_type, mode_dir=fold_dir, mode_name=recording['makam'])
			end_time = time.time()
			elapsed = (round((end_time - init_time) * 100) / 100)
			print elapsed
			output[('Fold' + str(fold))].append({'mbid':recording['mbid'], 'tonic_estimation':cur_out[0], 'sources': cur_out[1], 'distances':cur_out[2], 'elapsed_time':elapsed})
with open(os.path.join(distancePath, distance + '_k' + str(k_param) + '.json'), 'w') as f:
	json.dump(output, f, indent=2)
	f.close()