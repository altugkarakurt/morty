# -*- coding: utf-8 -*-
import json
import numpy as np
import os
import sys
sys.path.insert(0, './../')
import Evaluater as ev


#-----------------------------Parameters-----------------------------------
test_types = ['Joint', 'Tonic', 'Mode']
distance_list = ['bhat', 'intersection', 'corr', 'manhattan', 'euclidean', 'l3']
makam_list = ['Acemasiran', 'Acemkurdi', 'Beyati', 'Bestenigar', 'Hicaz', 'Hicazkar', 'Huseyni', 'Huzzam', 'Karcigar', 'Kurdilihicazkar', 
		      'Mahur', 'Muhayyer', 'Neva', 'Nihavent', 'Rast', 'Saba', 'Segah', 'Sultaniyegah', 'Suzinak', 'Ussak']
overall_tonic_list = []
#--------------------------------------------------------------------------

with open('annotations.json', 'r') as f:
	annot = json.load(f)
	f.close()

evaluater = ev.Evaluater()
experiment_dir = os.path.join('BozkurtExperiments')

for t in range(21,251):
	for test_type in test_types:
		print test_type
		training_dir = os.path.join(experiment_dir, 'Training' + str(t))
		test_dir = os.path.join(training_dir, test_type)

		for distance in distance_list:

			#Distance level initializations
			with open(os.path.join(training_dir, 'parameters.json'), 'r') as f:
				param = json.load(f)
				f.close()

			dist_result = {'folds':dict(), 'overall':dict(), 'parameters':param}
			if(test_type == 'Joint'):

				#Parameters for Tonic Histogram
				tonic_ss = 25
				tonic_diff_edges = np.arange(-tonic_ss/2.0, 1200+(tonic_ss/2.0), tonic_ss)
				tonic_bins = np.arange(0, (1200+tonic_ss), tonic_ss).tolist()[:-1]
				fold_tonic_list = []
				
				dist_result['overall'] = {'tonic_accuracy':0, 'mode_accuracy':0, 'joint_accuracy':0, 'tonic_histogram_bins':tonic_bins, 'tonic_histogram_vals':[], 'confusion':[[0 for i in range(20)] for j in range(20)], 'makam_list':makam_list}
			
			elif(test_type == 'Mode'):
				dist_result['overall'] = {'mode_accuracy':0, 'confusion':[[0 for i in range(20)] for j in range(20)]}

			elif(test_type == 'Tonic'):

				#Parameters for Tonic Histogram
				tonic_ss = 25
				tonic_diff_edges = np.arange(-tonic_ss/2.0, 1200+(tonic_ss/2.0), tonic_ss)
				tonic_bins = np.arange(0, (1200+tonic_ss), tonic_ss).tolist()[:-1]
				fold_tonic_list = []

				dist_result['overall'] = {'tonic_accuracy':0, 'tonic_histogram_bins':tonic_bins, 'tonic_histogram_vals':[]}
			
			#Retrieve the raw results
			with open(os.path.join(training_dir, test_type, (distance+'.json'))) as f:
				cur_test = json.load(f)
				f.close()

			for fold in range(1,11):
				cur_fold = cur_test[('Fold' + str(fold))]

				#Fold level initializations
				if(test_type == 'Joint'):
					dist_result['folds'][('Fold' + str(fold))] = {'tonic_accuracy':0, 'mode_accuracy':0, 'joint_accuracy':0, 'tonic_histogram_vals':[], 'confusion':[[0 for i in range(20)] for j in range(20)]}
				elif(test_type == 'Mode'):
					dist_result['folds'][('Fold' + str(fold))] = {'mode_accuracy':0, 'confusion':[[0 for i in range(20)] for j in range(20)]}
				elif(test_type == 'Tonic'):
					dist_result['folds'][('Fold' + str(fold))] = {'tonic_accuracy':0, 'tonic_histogram_vals':[]}
				dist_result['folds'][('Fold' + str(fold))]['individual'] = []

				#Iteration over each individual recording
				for k in cur_fold:
					cur_mbid = k['mbid']

					for j in annot:

						#retrieve annotated info about current recording
						if(cur_mbid == j['mbid']):
							cur_tonic = j['tonic']
							cur_mode = j['makam']
							break

					#Joint Estimation
					if(test_type == 'Joint'):
						est_mode = k['joint_estimation'][0][0]
						est_tonic = k['joint_estimation'][0][1]
						dist_result['folds'][('Fold' + str(fold))]['confusion'][makam_list.index(cur_mode)][makam_list.index(est_mode)] += 1
						dist_result['overall']['confusion'][makam_list.index(cur_mode)][makam_list.index(est_mode)] += 1
						cur_eval = evaluater.joint_evaluate(cur_mbid, (est_tonic, cur_tonic), (est_mode, cur_mode))
						fold_tonic_list.append(cur_eval['cent_diff'])
						overall_tonic_list.append(cur_eval['cent_diff'])
						
						if(cur_eval['mode_eval']):
							dist_result['folds'][('Fold' + str(fold))]['mode_accuracy'] += 1
							dist_result['overall']['mode_accuracy'] += 1

						if(cur_eval['tonic_eval']):
							dist_result['folds'][('Fold' + str(fold))]['tonic_accuracy'] += 1
							dist_result['overall']['tonic_accuracy'] += 1

						if(cur_eval['joint_eval']):
							dist_result['folds'][('Fold' + str(fold))]['joint_accuracy'] += 1
							dist_result['overall']['joint_accuracy'] += 1

					#Mode Estimation
					elif(test_type == 'Mode'):
						est_mode = k['tonic_estimation'][0]
						cur_eval = evaluater.mode_evaluate(cur_mbid, est_mode, cur_mode)
						dist_result['folds'][('Fold' + str(fold))]['confusion'][makam_list.index(cur_mode)][makam_list.index(est_mode)] += 1
						dist_result['overall']['confusion'][makam_list.index(cur_mode)][makam_list.index(est_mode)] += 1
						if(cur_eval['mode_eval']):
							dist_result['folds'][('Fold' + str(fold))]['mode_accuracy'] += 1
							dist_result['overall']['mode_accuracy'] += 1

					#Tonic Estimation
					elif(test_type == 'Tonic'):
						es_tonic = k['tonic_estimation'][0]
						cur_eval = evaluater.tonic_evaluate(cur_mbid, est_tonic, cur_tonic)
						fold_tonic_list.append(cur_eval['cent_diff'])
						overall_tonic_list.append(cur_eval['cent_diff'])

						if(cur_eval['tonic_eval']):
							dist_result['folds'][('Fold' + str(fold))]['tonic_accuracy'] += 1
							dist_result['overall']['tonic_accuracy'] += 1

					dist_result['folds'][('Fold' + str(fold))]['individual'].append(cur_eval)

				for key in list(set(dist_result['folds'][('Fold' + str(fold))].keys()) - set(['confusion', 'makam_list', 'individual', 'tonic_histogram_vals'])):
					dist_result['folds'][('Fold' + str(fold))][key] /= 100.0
				
				#tmp is useless and won't be used elsewhere
				dist_result['folds'][('Fold' + str(fold))]['tonic_histogram_vals'], tmp = np.histogram(fold_tonic_list, bins=tonic_diff_edges, density=False)
				dist_result['folds'][('Fold' + str(fold))]['tonic_histogram_vals'] = dist_result['folds'][('Fold' + str(fold))]['tonic_histogram_vals'].tolist()

			for key in list(set(dist_result['overall'].keys()) - set(['confusion', 'makam_list', 'tonic_histogram_bins', 'tonic_histogram_vals'])):
				dist_result['overall'][key] = round(dist_result['overall'][key] * 1) / 1000
			dist_result['overall']['tonic_histogram_vals'], tmp = np.histogram(fold_tonic_list, bins=tonic_diff_edges, density=False)
			dist_result['overall']['tonic_histogram_vals'] = dist_result['overall']['tonic_histogram_vals'].tolist()

			sys.exit()
			with open(os.path.join(test_dir, (distance+'_eval.json')), 'w') as f:
				json.dump(dist_result, f, indent=2) 
			print  distance + ' distance done!'
		print test_type + ' test_type done!'
	print 'Training ' + str(t) + ' done!'