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
distribution_type_list = ['pcd', 'pd']
chunk_size_list = [30, 60, 90, 120, 0]
makam_list = ['Acemasiran', 'Acemkurdi', 'Beyati', 'Bestenigar', 'Hicaz', 
			  'Hicazkar', 'Huseyni', 'Huzzam', 'Karcigar', 'Kurdilihicazkar', 
			  'Mahur', 'Muhayyer', 'Neva', 'Nihavent', 'Rast', 'Saba', 
			  'Segah', 'Sultaniyegah', 'Suzinak', 'Ussak']
arr = [1, 12, 45, 49, 50, 129, 133, 134, 145, 221, 225, 226, 237, 262, 265, 266, 267, 272, 275, 276, 277, 278, 279, 282, 288, 289, 296, 299, 313, 315, 317, 318, 319, 321, 322, 324, 325, 326, 329, 334, 337, 339, 348, 383, 384, 385, 391, 396, 411, 418, 419, 428, 429, 445, 450, 466, 467, 470, 471, 476, 481, 485, 494, 495, 500, 512, 519, 522, 523, 537, 539, 541, 543, 544, 563, 565, 572, 581, 585, 588, 593, 623, 639, 646, 647, 667, 675, 683, 690, 697, 711, 720, 730, 738, 767, 778, 783, 791, 799, 818, 831, 837, 843, 846, 850, 866, 870, 873, 874, 882, 892, 894, 898, 927, 935, 936, 984, 993, 1029, 1038, 1040, 1046, 1048, 1079, 1080, 1083, 1084, 1089, 1097, 1098, 1114, 1117, 1125, 1129, 1133, 1135, 1138, 1146, 1169, 1171, 1176, 1191, 1196, 1213, 1222, 1237, 1248, 1280, 1284, 1292, 1333, 1344, 1346, 1349, 1362, 1368, 1388, 1392, 1393, 1413, 1424, 1428, 1443, 1445, 1447, 1450, 1468, 1471, 1472, 1478, 1490, 1499, 1526, 1529, 1532, 1539, 1544, 1571, 1586, 1595, 1596, 1598, 1613, 1617, 1624, 1642, 1649, 1666, 1674, 1675, 1683, 1687, 1699, 1714, 1722, 1748, 1769, 1777, 1786, 1793, 1799, 1818, 1827, 1834, 1836, 1845, 1847, 1863, 1865, 1872, 1876, 1881, 1884, 1891, 1895, 1896, 1914, 1920, 1922, 1923, 1933, 1934, 1945, 1946, 1947, 1963, 1964, 1968, 1977, 1979, 1995, 1998, 2014, 2018, 2023, 2029, 2040, 2041, 2046, 2073, 2077, 2084, 2096, 2112, 2113, 2115, 2116, 2118, 2120, 2122, 2126, 2132, 2134, 2138, 2150, 2167, 2174, 2176, 2180, 2181, 2197, 2198, 2215, 2228, 2230, 2231, 2262, 2292, 2295, 2321, 2332, 2334, 2344, 2365, 2378, 2382, 2396, 2398, 2413, 2430, 2433, 2437, 2439, 2441, 2442, 2447, 2448, 2461, 2469, 2475, 2476, 2480, 2490, 2494, 2499]
x = arr[int(sys.argv[1])]

#data_folder = '../../../Makam_Dataset/Pitch_Tracks/'
#data_folder = '../../../test_datasets/turkish_makam_recognition_dataset/data/' #sertan desktop local
data_folder = '../../../experiments/turkish_makam_recognition_dataset/data/' # hpc cluster

# get the training experient/fold parameters 
idx = np.unravel_index(int(x), (len(fold_list), len(cent_ss_list), len(smooth_factor_list), len(distribution_type_list), len(chunk_size_list)))
fold = fold_list[idx[0]]
cent_ss = cent_ss_list[idx[1]]
smooth_factor = smooth_factor_list[idx[2]]
distribution_type = distribution_type_list[idx[3]]
chunk_size = chunk_size_list[idx[4]]

# instantiate makam estimator for training
estimator = be.BozkurtEstimation(cent_ss=cent_ss, smooth_factor=smooth_factor, chunk_size=chunk_size)

# experiment info
experiment_info = {'cent_ss': cent_ss, 'smooth_factor':smooth_factor, 'distribution_type':distribution_type, 'chunk_size':chunk_size, 'method':'bozkurt'}

# folder structure
experiment_master_dir = './Experiments' # assumes it is already created

experiment_dir = os.path.join(experiment_master_dir, 'Experiment' + str(x))
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
	print 'Already done training: ' + str(x)
	sys.exit()

# load annotations; the tonic values will be read from here
with open('annotations.json', 'r') as f:
	annot = json.load(f)
	f.close()

# load the fold to get the training recordings
with open((os.path.join('./Folds', 'fold_' + str(fold) + '.json')), 'r') as f:
	cur_fold = json.load(f)['test']
	f.close()

print 'Starting training: ' + str(x)

# retrieve annotations of the training recordings
for makam_name in makam_list:
	# divide the training data into makams
	makam_annot = [k for k in cur_fold if k['makam']==makam_name]
	for i in makam_annot:
		for j in annot:
			# append the tonic of the recording from the relevant annotation
			if(i['mbid'] == j['mbid']):
				i['tonic'] = j['tonic'] 
				break

	# get pitch tracks and tonic frequencies
	pitch_track_dir = os.path.join(data_folder, makam_name)
	pitch_track_list = [(ma['mbid'] + '.pitch') for ma in makam_annot]
	tonic_list = [ma['tonic'] for ma in makam_annot]

	# train
	estimator.train(makam_name, pitch_track_list, tonic_list, metric=distribution_type_list, 
		pt_dir=pitch_track_dir, save_dir=fold_dir)

print '   Finished training: ' + str(x)

