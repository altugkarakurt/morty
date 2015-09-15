# Mode & Tonic Recognition
Python scripts for training and recognizing modes in a modal music piece. Given the predominant melody of a piece, it estimates its mode and tonic.

### Description and Usage
This project expects the pitch track (or predominant melody) of audio recordings as the input and generates pitch distributions (PD) and
pitch class distributions (PCD) from them. These distributions are used as the parameters for estimation and training.

The algorithms can be used for both **estimating tonic and mode of a piece**. However, if either is known and this information could be
fed into the system, and hence the estimation of other would be more accurate.

For the estimation:
* Train the candidate modes by using the collections of pitch tracks of respective modes.
* Feed the piece's pitch track and if any the known attribute (tonic or mode) into the system and you're done.

If the pitch track of a piece isn't available, the pitch.py script in [pycompmusic](https://github.com/mtg/pycompmusic)
project's pycompmusic-master/compmusic/extractors/makam directory can be used to generate it. The pitch track is expected
to be in given as a .txt file, that consists of a single column of values of the pitch track in Hertz, time information
isn't required.

Since the training a supervised machine learning process, a dataset for each mode, including pieces with annotated tonic frequencies, is preliminary.

### Dependencies
This project depends on [NumPy](http://www.numpy.org/), [SciPy](http://www.scipy.org/), [Matplotlib](http://matplotlib.org/) and [Essentia](https://github.com/MTG/essentia).

### Explanation of Classes
* *PitchDistribution* is the class, which holds the pitch distribution. It also includes save and load functions to make the pitch distributions accessible for later use.

* *BozkurtEstimation* implements the methods proposed in (A. C. Gedik, B.Bozkurt, 2010) and (B. Bozkurt, 2008).

* *ChordiaEstimation* implements the method proposed in (Chordia, P. and Şentürk, S. 2013).

* *ModeFunctions* includes the low-level functions related to mode and tonic recognition. These functions are generic and common in both Bozkurt and Chordia methods.
They aren't expected to be used directly; instead they are called by the higher level wrapper functions in BozkurtEstimation and ChordiaEstimation.

### References

> A. C. Gedik, B.Bozkurt, 2010, "Pitch Frequency Histogram Based Music Information Retrieval for Turkish Music", Signal Processing, vol.10,

> B. Bozkurt, 2008, "An automatic pitch analysis method for Turkish maqam music", Journal of New Music Research 37 1–13.

> Chordia, P. and Şentürk, S. (2013). Joint recognition of raag and tonic in North Indian music. Computer Music Journal, 37(3):82–98.