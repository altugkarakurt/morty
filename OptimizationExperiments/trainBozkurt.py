# -*- coding: utf-8 -*-
from datetime import datetime
import numpy as np
import sys
import json
import os
from os import path
from generate_ten_fold import load_fold
sys.path.insert(0, './../')
import BozkurtEstimation as be

def train(makam_name, pt_list, tonic_list, cent_ss, smooth_factor, metric, method, pt_dir='./', fold_dir='./', chunk_size=60):
	### Bozkurt Estimation
	b = be.BozkurtEstimation(cent_ss=cent_ss, smooth_factor=smooth_factor, chunk_size=chunk_size)
	b.train(makam_name, pt_list, tonic_list, metric=metric, pt_dir=pt_dir, save_dir=fold_dir)

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

# get the training experient/fold parameters 
idx = np.unravel_index(int(sys.argv[1]), (len(fold_list), len(cent_ss_list), len(smooth_factor_list), len(metric_list), len(chunk_size_list)))
fold = fold_list[idx[0]]
cent_ss = cent_ss_list[idx[1]]
smooth_factor = smooth_factor_list[idx[2]]
metric = metric_list[idx[3]]
chunk_size = chunk_size_list[idx[4]]

# experiment info
experiment_info = {'cent_ss': cent_ss, 'smooth_factor':smooth_factor, 'metric':metric, 'chunk_size':chunk_size, 'method':'bozkurt'}

# folder structure
experiment_master_dir = './Experiments' # assumes it is already created

experiment_dir = os.path.join(experiment_master_dir, 'Experiment' + sys.argv[1])
if not os.path.exists(experiment_dir):
	os.makedirs(experiment_dir)

fold_dir = os.path.join(experiment_dir, 'Fold' + str(fold))
if not os.path.exists(fold_dir):
	os.makedirs(fold_dir)


'''
print 'Here we go! ' + str(datetime.now())

with open('annotations.json', 'r') as f:
	annot = json.load(f)
	f.close()

###Experiment------------------------------------------------------------------------------------
for cent_ss in cent_ss_list:						#Bozkurt
	for smooth_factor in smooth_factor_list:
		for metric in metric_list:
			for chunk_size in chunk_size_list:
				cnt += 1
				if(os.path.isdir('./Experiments/Experiment' + str(cnt+1))):	#The trial is completed
					continue
				elif(os.path.isdir('./Experiments/Experiment' + str(cnt))): 
					if(os.path.isdir('./Experiments/Experiment' + str(cnt) + '/Fold10')):
						continue
					else:
						min_fld = 1 + len(os.listdir('./Experiments/Experiment' + str(cnt)))
				else:
					min_fld = 1
					print 'Experiment ' + str(cnt) + ' Started\tMetric: ' + metric + ', Smooth_Factor: ' + str(smooth_factor) + ', Cent_SS: ' + str(cent_ss) + ', Chunk Size: ' + str(chunk_size)
					os.makedirs('./Experiments/' + 'Experiment' + str(cnt) + '/')
				for fld in np.arange((min_fld),(max_fld+1)):
					cur_fold = load_fold('./Folds/fold_' + str(fld) + '.json')['train']
					fold_dir = './Experiments/' + 'Experiment' + str(cnt) + '/Fold' + str(fld) + '/'
					os.makedirs(fold_dir)
					for makam_name in makam_list:
						makam_annot = []
						tmp_annot = [k for k in cur_fold if (k.values()[0]==makam_name)]
						for i in tmp_annot:
							for j in annot:
								if(i.keys()[0] == j['mbid']):
									makam_annot.append(j)
									break
						pt_dir = '../../../test_datasets/turkish_makam_recognition_dataset/data/' + makam_name + '/'			
						pt_list = [(tmp['mbid'] + '.pitch') for tmp in makam_annot]
						tonic_list = [tmp['tonic'] for tmp in makam_annot]
						train(makam_name, pt_list, tonic_list, cent_ss, smooth_factor, metric, 'bozkurt', pt_dir=pt_dir, fold_dir=fold_dir, chunk_size=chunk_size)
					experiment_info = {'cent_ss': cent_ss, 'smooth_factor':smooth_factor, 'metric':metric, 'chunk_size':chunk_size, 'method':'bozkurt'}
					with open('experiment_info.json', 'a') as f:
						json.dump(experiment_info, f, indent=2)
						f.close()
					print 'Fold ' + str(fld) + ' Done! ' + str(datetime.now())
				print 'Experiment ' + str(cnt) + ' Done\tMetric: ' + metric + ', Smooth_Factor: ' + str(smooth_factor) + ', Cent_SS: ' + str(cent_ss) + ', Chunk Size: ' + str(chunk_size) + ' \n'
'''