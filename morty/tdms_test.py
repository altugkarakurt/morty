import numpy as np

from converter import Converter
from tdms import TDMS

hz_track = np.loadtxt("00f1c6d9-c8ee-45e3-a06f-0882ebcb4e2f.pitch")
tonic = 256.0
tau = 0.3
cent_track = Converter.hz_to_cent(hz_track, tonic)
surface = TDMS.from_cent_pitch(cent_track, tau, ref_freq=tonic, step_size=120)
