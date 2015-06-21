# Mode & Tonic Recognition
Python scripts for training and recognizing modes in a modal music piece. Given the predominant melody of a piece, it estimates its mode and tonic.

### Description and Usage
This project expects the pitch track, or predominant melody, as the input and generates pitch distributions (PD) and
pitch class distributions (PCD) from them. These distributions are used as the parameters for estimation and training
processes.

It can be used for **estimating tonic and/or mode of a piece**. However, if either is known and this information would be
fed into the system, the estimation of other would be more accurate.

For the estimation:
* Train the candidate modes by using the collections of pitch tracks of respective modes.
* Feed the piece's pitch track and if any the known attribute (tonic or mode) into the system and you're done.

If the pitch track of a piece isn't available, the pitch.py script in [pycompmusic](https://github.com/mtg/pycompmusic)
project's pycompmusic-master/compmusic/extractors/makam directory can be used to generate it. The pitch track is expected
to be in given as a .txt file, that consists of a single column of values of the pitch track in Hertz, time information
isn't required.

Since the training portion of the process is a supervised machine learning, a dataset for each mode, including pieces with known tonics, is a preliminary.

### Dependencies
This project depends on [NumPy](http://www.numpy.org/), [SciPy](http://www.scipy.org/), [Matplotlib](http://matplotlib.org/) and [Essentia](https://github.com/MTG/essentia).

### Explanation of Classes
* *PitchDistribution* is used the data structure of pitch distributions. It wraps the related data to a pitch
distribution. It also includes save and load functions to make the pitch distributions accessible for later use.

* *BozkurtEstimation* is the most important class of the project. It wraps the other two classes and provides higher
level functions that the users are expected to interact. The training and estimation functions are in this class.

* *MakamFunctions* includes the low-level functions related to the task. These functions aren't going to be directly
used, they are called by the higer level functions in Bozkurt Estimation.

### Examples
As an example, the pitch tracks of three Ottoman-Turkish Makam Music piece are given in the project, along with their
PD, PCD and the joint PD and PCD of their common makam (Turkish mode) Ussak. These are for demonstrating the structure
of the data and to be used for discovering the project's functionalities.

Example test functions based on this data will be added soon.

### Related Literature
To be continued...

