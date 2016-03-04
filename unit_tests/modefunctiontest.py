import numpy as np
import sys

sys.path.append('../modetonicestimation')
import ModeFunctions as mf
from PitchDistribution import PitchDistribution

def test_generate_pcd:
	pitch_track = np.loadtxt('pitch_track.txt')
	pd = mf.generate_pd(pitch_track)
	pcd = pd = mf.generate_pcd(pd)
	test_pd = PitchDistribution.load('test_pd')
	test_pcd = PitchDistribution.load('test_pcd')
	assert ((pd == test_pd) and (pcd == test_pcd))