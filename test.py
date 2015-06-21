import PitchDistribution as p_d
import ModeFunctions as mf
import BozkurtEstimation as be
import matplotlib.pyplot as pl
import numpy as np

"""---------------------------------------------------------------------------------------
This is an example usage of the functions. 

* Since both pitch distributions (PDs) and pitch class distributions (PCDs) are saved as
PitchDistribution objects, the functions and the  attributes can be called/accessed the
same way. The distinguishment of these are handled internally.
---------------------------------------------------------------------------------------"""

###---------------------------------------------------------------------------------------

### Initializations
pt_dir = 'Examples/Pitch Tracks/'
pd_dir = 'Examples/PD/'
pcd_dir = 'Examples/PCD/'
b = be.BozkurtEstimation()

###---------------------------------------------------------------------------------------

### Loading the pitch tracks
pt1 = mf.load_track('semahat', pt_dir)[:,1]


###---------------------------------------------------------------------------------------

### Loading the existing pitch distributions. The JSON related issues are handled 
### internally, no need to import json.
pcd1 = p_d.load('semahat_pcd.json', pcd_dir)
pcd2 = p_d.load('gec_kalma_pcd.json', pcd_dir)
pcd3 = p_d.load('murat_derya_pcd.json', pcd_dir)
### You don't need to worry about KDE, if you just want to use the function as it is. KDE
### returns the Kernel Density Estimation, in case you might use in another analysis.
pd = p_d.load('gec_kalma_pd.json', pd_dir)

### They can plotted like this.
#pcd1.plot() # This is Figure 1
#pd.plot() # This is Figure 2

###---------------------------------------------------------------------------------------

### Here comes the actual training part. After the following lines, the joint distributions
### of the modes should be saved in your working directory.
ussak_pcd = b.train('ussak_pcd', [(pt_dir + 'semahat'), (pt_dir + 'gec_kalma'), (pt_dir + 'murat_derya')], [199, 396.3525, 334.9488], metric='pcd')
ussak_pd = b.train('ussak_pd', [(pt_dir + 'semahat'), (pt_dir + 'gec_kalma'), (pt_dir + 'murat_derya')], [199, 396.3525, 334.9488], metric='pd')

### Let's see if the joint PCD is similar to the marginal PCDs. Blue is the joint PCD. The
### yellow ones are the marginals. I am using matplotlib to modify the colors.
"""pl.figure() 
pl.plot(ussak_pcd.vals, 'b')
pl.plot(pcd1.vals, 'y')
pl.plot(pcd2.vals, 'y')
pl.plot(pcd3.vals, 'y')
pl.show() # This is Figure 3 
"""
###---------------------------------------------------------------------------------------

### Training is completed. It's time to estimate. Since there aren't any candidate modes, I
### am treating the PCDs of other pieces as joint PCDs of a different mode.

m = b.estimate(pt1, mode_names=[], mode_name='semahat', est_tonic=True, est_mode=False, rank = 3, mode_dir=pcd_dir, distance_method="euclidean", metric='pcd', ref_freq=440)
print m
### Soon the joint estimation function will be added. In that case neither the tonic nor the
### mode would be known and the function would estimate both.
# TODO

###---------------------------------------------------------------------------------------


"""---------------------------------------------------------------------------------------
The following part is for demonstrating the usage of low-level functions. Unless you want
to change the algorithm or need the intermediate values or steps you might not need these.
---------------------------------------------------------------------------------------"""
"""
###---------------------------------------------------------------------------------------

### Generation of pitch distributions (PD), using the known tonics. For kde*, see line 37.
pd1, kde1 = mf.generate_pd(pt1, ref_freq=199)
pd2, kde2 = mf.generate_pd(pt2, ref_freq=396.3525)
pd3, kde3 = mf.generate_pd(pt3, ref_freq=334.9488)

### Generation of pitch class distributions (PCD) from the generated PDs
pcd1 = mf.generate_pcd(pd1)
pcd2 = mf.generate_pcd(pd2)
pcd3 = mf.generate_pcd(pd3)

### You can save them like this.
pcd1.save('test_save.json')

### To Be Continued...

print 'Done!'

###---------------------------------------------------------------------------------------"""