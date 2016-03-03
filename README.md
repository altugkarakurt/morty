# Mode & Tonic Recognition
Python scripts for training and recognizing modes in a modal music piece. Given the predominant melody of a piece, it estimates its mode and tonic.

### Usage
Please refer to the jupyter notebooks for the basic usage and an interactive demo.

### Description
This project expects the pitch track (or predominant melody) of audio recordings as the input and generates pitch distributions (PD) and
pitch class distributions (PCD) from them. These distributions are used as the parameters for estimation and training.

The algorithms can be used for both **estimating tonic and mode of a piece**. However, if either is known and this information could be
fed into the system, and hence the estimation of other would be more accurate.

For the estimation:
* Train the candidate modes by using the collections of pitch tracks of respective modes.
* Feed the piece's pitch track and if any the known attribute (tonic or mode) into the system and you're done.

If the pitch track of a piece isn't available, [melodyExtraction.py](https://github.com/altugkarakurt/ModeTonicEstimation/blob/master/extras/melodyExtraction.py) function in the extras package can be used to obtain the pitch tracks automatically. The pitch track is expected to be in given as a .txt file, that consists of a single column of values of the pitch track in Hertz, time information isn't required. This function is a wrapper around the [predominantmelodymakam](https://github.com/sertansenturk/predominantmelodymakam) package to store the pitch track in the desired format.

Since the training a supervised machine learning process, a dataset for each mode, including pieces with annotated tonic frequencies, is preliminary.

### Installation

If you want to install the repository, it is recommended to install the package and dependencies into a virtualenv. In the terminal, do the following:

    virtualenv env
    source env/bin/activate
    python setup.py install

If you want to be able to edit files and have the changes be reflected, then
install the repository like this instead

    pip install -e .

The algorithm uses several modules in Essentia. Follow the [instructions](essentia.upf.edu/documentation/installing.html) to install the library.

Now you can install the rest of the dependencies:

    pip install -r requirements

For the functionalities in extras package, you can install the optional dependencies as:

    pip install -r optional_requirements

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

> Atlı, H. S., Uyar, B., Şentürk, S., Bozkurt, B., and Serra, X. (2014). Audio feature extraction for exploring Turkish makam music. In Proceedings of 3rd International Conference on Audio Technologies for Music and Media, Ankara, Turkey.
