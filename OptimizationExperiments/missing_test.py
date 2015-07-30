# -*- coding: utf-8 -*-
import json
import numpy as np
import os
import sys
import testBozkurt_tonic as tonic
import testBozkurt as joint
import testBozkurt_mode as mode
sys.path.insert(0, './../')
import FileOperations as fo

experiment_dir = 'BozkurtExperiments'

i = int(sys.argv[1]) - 1

folder_list = ['Joint', 'Mode', 'Tonic']
file_list = ['bhat', 'intersection', 'euclidean', 'manhattan', 'l3', 'corr']

idx = np.unravel_index(i, (len(folder_list), len(file_list), len(range(1,251))))

folder = folder_list[idx[0]]
distance = file_list[idx[1]]
training_idx = range(1,251)[idx[2]]
print folder
print distance
print training_idx
training_dir = os.path.join(experiment_dir, ('Training' + str(training_idx)))
dist_dir = os.path.join(training_dir, folder)

if folder == 'Mode':
	mode.run(distance, training_idx)
elif folder == 'Tonic':
	tonic.run(distance, training_idx)
elif folder == 'Joint':
	joint.run(distance, training_idx) 