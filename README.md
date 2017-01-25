[![GitHub version](https://badge.fury.io/gh/altugkarakurt%2Fmorty.svg)](https://badge.fury.io/gh/altugkarakurt%2Fmorty) 
[![Build Status](https://travis-ci.org/altugkarakurt/morty.svg?branch=master)](https://travis-ci.org/altugkarakurt/morty) [![Code Climate](https://codeclimate.com/github/altugkarakurt/ModeTonicEstimation/badges/gpa.svg)](https://codeclimate.com/github/altugkarakurt/ModeTonicEstimation) [![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-ff69b4.svg)](http://www.gnu.org/licenses/agpl-3.0)

### morty
**MO**de **R**ecognition and **T**onic **Y**dentification Toolbox:

### Introduction
**morty** is a toolbox for mode recognition and tonic identification in audio performances of "modal" music cultures. The toolbox is based on well-studied pitch histogram analysis. It implements two state of the art methods applied to Ottoman-Turkish makam music (A. C. Gedik and B.Bozkurt, 2010) and Hindustani music (P. Chordia and S. Şentürk, 2013).

Please cite the publication below, if you use the toolbox in your work:

> Karakurt, A., Şentürk S., & Serra X. (2016).  [MORTY: A Toolbox for Mode Recognition and Tonic Identification](http://mtg.upf.edu/node/3538). 3rd International Digital Libraries for Musicology Workshop. New York, USA

The main purpose of the toolbox is to provide a quick access to automatic tonic identification and mode recognition implementations for music cultures, for which these tasks have not been addressed and provide a baseline for novel methodologies to be compared against. Therefore, the implementations are designed such that there is no "computational-bias" towards a particular culture yet any culture-specific optimization can be easily introduced in the context of the implemented methodologies. 

The pitch distribution and pitch class distributions implemented in this package can be additionally used for other relevant tasks such as [tuning analysis](https://github.com/miracatici/notemodel), [intonation-analysis](https://github.com/sertansenturk/alignednotemodel) and [melodic progression analysis](https://github.com/sertansenturk/seyiranalyzer). Furthermore the applied analysis can be used in cross-cultural comparisons.

### Description
The methodologies proposed in (A. C. Gedik and B.Bozkurt, 2010) and (P. Chordia and S. Şentürk, 2013) are based on the musical assumption that the tuning and the relative occurence of the melodic intervals in the performances belonging to the same mode should also be similar.

Given the annotated tonics and makams for a set of training audio performances, both methods extract predominant melody of each performance and then compute models based on pitch histograms (pitch distributions or pitch-class distributions) for each mode using the extracted predominant melodies. Note that the training performances can be entire recordings or an excerpt.

In our context, these models are used in three similar computational tasks:
- **Mode Recognition:** Given an audio performance with known tonic, the pitch histogram computed from the performance is compared with the model produced for each mode. The mode belonging to the most similar model will be classified as the estimated mode.
- **Tonic Identification:** Given an audio performance with known mode, the pitch histogram computed from the performance is shifted and compared with the model of the mode. The shift that produces the highest similarity will indicate the estimated tonic.
- **Joint Estimation:** Given an audio performance with unknown tonic and mode, the pitch histogram computed from the performance is shifted and compared with the model of each mode. The most similar shift and the mode of the matching model yields the estimated tonic and the mode jointly.

For an indepth explanation of the concept and the methodologies, please refer to the papers.

### Usage
This algorithms expect the predominant melody of the audio performances as the input and generates pitch distributions (PD) or pitch class distributions (PCD) from them. These distributions are used as the features for the training and the estimation.

The algorithms can be used for both **estimating tonic and mode of a piece**. When either of these two is available and this information could be fed into the algorithm, and hence the estimation would be more accurate.

Since the training step is a supervised machine learning process, a dataset for each mode, including audio with annotated tonic frequencies, is preliminary. Basically the steps for the methodologies are:
* Train the candidate modes by using the collections of predominant melodies of respective modes extracted from the annotated audio.
* Feed the predominant melody of the testing audio with the known attributes (tonic or mode), if any, and obtain the estimation(s).

Please refer to the jupyter notebooks in the [demos](https://github.com/altugkarakurt/morty/tree/master/demo) folder for the basic usage.

If the predominant melodies are not available, [melodyextraction.py](https://github.com/altugkarakurt/morty/blob/master/extras/melodyExtraction.py) method in the extras package can be used for automatic predominant melodies extraction. This method is a wrapper [implementation](https://github.com/sertansenturk/predominantmelodymakam) of the predominant melody extraction methodology proposed by Atlı et. al (2014) to store the pitch track in the desired format. The input pitch track is expected to be in given as a .txt file, that consists of a single column of values of the pitch track in Hertz. The timestamps are not required. Note that the default parameters for predominant melody extraction are optimized for Ottoman-Turkish makam music, so you might want to calibrate the parameters according to the necessities of the studied music culture.

### Installation

If you want to install the repository, it is recommended to install the package and dependencies into a virtualenv. In the terminal, do the following:

    virtualenv env
    source env/bin/activate
    python setup.py install

If you want to be able to edit files and have the changes be reflected, then install the repository like this instead

    pip install -e .

The algorithm uses several modules in Essentia. Follow the [instructions](essentia.upf.edu/documentation/installing.html) to install the library.

For the functionalities in the extras package, you can install the optional dependencies as:

    pip install -r optional_requirements

### Explanation of Classes
- *PitchDistribution* is the class, which holds the pitch distribution. It also includes save and load functions to make the pitch distributions accessible for later use.

- *KNNClassifier* class implements and generalizes the methods proposed in (A. C. Gedik, B. Bozkurt, 2010) and (B. Bozkurt, 2008) and (P. Chordia and S. Şentürk, 2013).

### References

> A. C. Gedik, B.Bozkurt, 2010, "Pitch Frequency Histogram Based Music Information Retrieval for Turkish Music", Signal Processing, vol.10,

> B. Bozkurt, 2008, "An automatic pitch analysis method for Turkish maqam music", Journal of New Music Research 37 1–13.

> P. Chordia. and S. Şentürk. (2013). Joint recognition of raag and tonic in North Indian music. Computer Music Journal, 37(3):82–98.

> H. S. Atlı, B. Uyar, S. Şentürk, B. Bozkurt, and X. Serra (2014). Audio feature extraction for exploring Turkish makam music. In Proceedings of 3rd International Conference on Audio Technologies for Music and Media, Ankara, Turkey.
