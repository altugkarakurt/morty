# -*- coding: utf-8 -*-
import numpy as np
import sys
import json
import os
from os import path
import generate_ten_fold as fold
sys.path.insert(0, './../')
import BozkurtEstimation as be
import ModeFunctions as mf



###Experiment Parameters-------------------------------------------------------------------------
distance_list = ['manhattan', 'euclidean', 'l3', 'bhat', 'intersection', 'corr']
est_param = [(True, True), (True, False), (False, True)] #(est_mode, est_tonic)
cent_ss_list = [7.5, 15, 25, 50, 100]
smooth_factor_list = [7.5, 10, 15, 20, 2.5]
metric_list = ['pcd', 'pd']
chunk_size_list = [30, 60, 90, 120]
makam_list = ['Acemasiran', 'Acemkurdi', 'Beyati', 'Bestenigar', 'Hicaz', 'Hicazkar', 'Huseyni', 'Huzzam', 'Karcigar', 'Kurdilihicazkar', 'Mahur', 'Muhayyer', 'Neva', 'Nihavent', 'Rast', 'Saba', 'Segah', 'Sultaniyegah', 'Suzinak', 'Ussak']


rank = 5
trial_info = dict()
cnt = 0
max_fld=10
min_fld=1
print 'Here we go! ' + str(datetime.now())
###Experiment------------------------------------------------------------------------------------
for distance in distance_list:
	for param in est_param:
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
							print 'Experiment ' + str(cnt) + ' Started\tMetric: ' + metric + ', Smooth_Factor: ' + str(smooth_factor) + ', Cent_SS: ' + str(cent_ss) + ', Chunk Size: ' + str(chunk_size)
							os.makedirs('./Experiments/' + 'Experiment' + str(cnt) + '/')
						for fld in np.arange((min_fld),(max_fld+1)):
							cur_fold = fold.load_fold('./Experiments/Folds/train_fold_' + str(fld) + '.json')
							save_dir = './Experiments/' + 'Experiment' + str(cnt) + '/Fold' + str(fld) + '/'
							os.makedirs(save_dir)
							for makam_name in makam_list:
								makam_annot = [k for k in cur_fold if (k['makam']==makam_name)]
								pt_dir = '../../../Makam_Dataset/Pitch_Tracks/' + makam_name + '/'			
								pt_list = [(tmp['mbid'] + '.pitch') for tmp in makam_annot]
								tonic_list = [tmp['tonic'] for tmp in makam_annot]
								train(makam_name, pt_list, tonic_list, cent_ss, smooth_factor, metric, 'bozkurt', pt_dir=pt_dir, save_dir=save_dir, chunk_size=chunk_size)
							print 'Fold ' + str(fld) + ' Done! ' + str(datetime.now())
						trial_info = {('Experiment' + str(cnt)):{'cent_ss': cent_ss, 'smooth_factor':smooth_factor, 'metric':metric, 'chunk_size':chunk_size, 'method':'bozkurt'}}
						with open('trial_info.json', 'a') as f:
							json.dump(trial_info, f, indent=2)
							f.close()
						print 'Experiment ' + str(cnt) + ' Done\tMetric: ' + metric + ', Smooth_Factor: ' + str(smooth_factor) + ', Cent_SS: ' + str(cent_ss) + ', Chunk Size: ' + str(chunk_size) + ' \n'



cur_fold = fold.load_fold('./Bozkurt_Experiment/bozkurt_test_fold.json')
save_dir = './Bozkurt_Experiment/'
b = be.BozkurtEstimation(cent_ss=cent_ss, smooth_factor=smooth_factor, chunk_size=chunk_size)
results = []
print 'Mode Estimation'
for makam_name in makam_list:
	makam_annot = [k for k in cur_fold if (k['makam']==makam_name)]
	pt_dir = '../../../Makam_Dataset/Pitch_Tracks/' + makam_name + '/'
	pt_list = [(tmp['mbid'] + '.pitch') for tmp in makam_annot]
	tonic_list = [tmp['tonic'] for tmp in makam_annot]
	for pt in range(len(pt_list)):
		print 'new track'
		pitch_track = mf.load_track(pt_list[pt], pt_dir)
		cur_res = b.estimate(pitch_track, mode_names=makam_list, mode_name='', mode_dir=save_dir, est_tonic=True, est_mode=True, rank=1, distance_method="euclidean", metric='pcd', ref_freq=tonic_list[pt])
		results.append({'mbid': pt_list[pt][:-6], 'makam':makam_name, 'tonic':tonic_list[pt], 'estimated':cur_res})