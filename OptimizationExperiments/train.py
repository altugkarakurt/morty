# -*- coding: utf-8 -*-

import generate_ten_fold as gen
import numpy as np
import sys
import json
import os
from os import path
import generate_ten_fold as fold

sys.path.insert(0, '/home/altug/Desktop/Work/MTG/git_dir/ModeRecognition')
import BozkurtEstimation as be
import ChordiaEstimation as che

sys.path.insert(0, '/home/altug/Desktop/Work/MTG/Makam_Dataset')
import FileOperations as fo

def train(makam_name, pt_list, tonic_list, cent_ss, smooth_factor, metric, method, pt_dir='./', save_dir='./', chunk_size=60, threshold=0.5, overlap=0):

	### Chordia Training
	if (method=='chordia'):
		c = che.ChordiaEstimation(cent_ss=cent_ss, smooth_factor=smooth_factor, chunk_size=chunk_size, threshold=threshold, overlap=overlap)
		c.train(makam_name, pt_list, tonic_list, metric='pcd', pt_dir=pt_dir, save_dir=save_dir)
	
	### Bozkurt Estimation
	elif(method=='bozkurt'):
		b = be.BozkurtEstimation(cent_ss=cent_ss, smooth_factor=smooth_factor, chunk_size=chunk_size)
		b.train(makam_name, pt_list, tonic_list, metric=metric, pt_dir=pt_dir, save_dir=save_dir)

###Experiment Parameters-------------------------------------------------------------------------
cent_ss_list = [7.5, 15, 25, 50, 100]
smooth_factor_list = [7.5, 10, 15, 20, 50]
metric_list = ['pcd', 'pd']
overlap_list = [0, 0.25, 0.50, 0.75]
chunk_size_list = [30, 60, 90, 120]
makam_list = ['Acemasiran', 'Acemkurdi', 'Beyati', 'Bestenigar', 'Hicaz', 'Hicazkar', 'Huseyni', 'Huzzam', 'Karcigar', 'Kurdilihicazkar', 'Mahur', 'Muhayyer', 'Neva', 'Nihavent', 'Rast', 'Saba', 'Segah', 'Sultaniyegah', 'Suzinak', 'Ussak']
trial_info = dict()
cnt = 0

###Experiment------------------------------------------------------------------------------------
for cent_ss in cent_ss_list:						#Bozkurt
	for smooth_factor in smooth_factor_list:
		for metric in metric_list:
			for chunk_size in chunk_size_list:
				cnt += 1
				os.makedirs('/home/altug/Desktop/Work/MTG/Experiment/' + 'Trial' + str(cnt) + '/')
				for fld in np.arange(1,11):
					cur_fold = fold.load_fold('/home/altug/Desktop/Work/MTG/Experiment/Folds/train_fold_' + str(fld) + '.json')
					save_dir = '/home/altug/Desktop/Work/MTG/Experiment/' + 'Trial' + str(cnt) + '/Fold' + str(fld) + '/'
					os.makedirs(save_dir)
					for makam_name in makam_list:
						makam_annot = [k for k in cur_fold if (k['makam']==makam_name)]
						pt_dir = '/home/altug/Desktop/Work/MTG/Makam_Dataset/Pitch_Tracks/' + makam_name + '/'			
						pt_list = [(tmp['mbid'] + '.pitch') for tmp in makam_annot]
						tonic_list = [tmp['tonic'] for tmp in makam_annot]
						train(makam_name, pt_list, tonic_list, cent_ss, smooth_factor, metric, 'bozkurt', pt_dir=pt_dir, save_dir=save_dir, chunk_size=chunk_size)
						trial_info[('Trial' + str(cnt))] = {'cent_ss': cent_ss, 'smooth_factor':smooth_factor, 'metric':metric, 'makam_name':makam_name}
					print 'Fold ' + fld + ' Done!'
				print 'Trial ' + str(cnt) + ': ' + metric + ', Smooth_Factor: ' + str(smooth_factor) + ', Cent_SS: ' + str(cent_ss) + ', Chunk Size: ' + str(chunk_size) + ' Done!\n'
"""
for cent_ss in cent_ss_list:						#Chordia
	for smooth_factor in smooth_factor_list:
		for metric in metric_list:
			for chunk_size in chunk_size_list:
				for overlap in overlap_list:
					for makam_name in makam_list:
						cnt += 1
						makam_annot = [k for k in annot if (k['makam']==makam_name)]
						save_dir = '/home/altug/Desktop/Work/MTG/Experiment/' + 'Trial' + str(cnt)
						os.makedirs(save_dir)	
						pt_dir = '/home/altug/Desktop/Work/MTG/Makam_Dataset/Pitch_Tracks/' + m_name
						pt_list = fo.getFileNamesInDir('/home/altug/Desktop/Work/MTG/Makam_Dataset/Pitch_Tracks', 'json')[2]
						mbid_list = [name[:-5] for name in pt_list]
						tonic_list = [d['tonic'] for mbid in mbid_list for d in makam_annot if (d['mbid']==mbid)]
						train(makam_name, pt_list, tonic_list, cent_ss, smooth_factor, metric, 'bozkurt', pt_dir='./', save_dir='./')
						trial_info[('Trial' + str(cnt))] = {'cent_ss': cent_ss, 'smooth_factor':smooth_factor, 'metric':metric, 'makam_name':makam_name}
						print str(cnt) + ': ' + metric + ', Smooth_Factor: ' + smooth_factor + ', Cent_SS: ' + cent_ss + ' Done!'

with open('parameters.json', 'a') as g:
	json.dump(trial_info, g, indent=2)
	g.close
"""