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

i = int(sys.argv[1])

training_dir = os.path.join(experiment_dir, ('Training' + str(i)))
for test_type in ['Joint', 'Mode', 'Tonic']:
	print "> Training " + str(i)
	dist_dir = os.path.join(training_dir, test_type)

	if test_type == 'Mode':
		print ">> Mode"
		mode.run(i)
	elif test_type == 'Tonic':
		print ">> Tonic"
		tonic.run(i)
	elif test_type == 'Joint':
		print ">> Joint"
		joint.run(i) 