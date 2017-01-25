#### morty v1.3.0
 - Added bar method to PitchDistribution
 - Improved PitchDistribution.plot 
 - Changed the default tonic deviation threshold from 25 cents to 20 cents (see: commit 7b385c)

#### morty v1.2.1
 - Removed the experimental neural networks from the master branch
 - Updated the readme to include the DLFM paper
 - Fixed PCD conversion and PCD checking when the input distribution's bins are in Hz
 - PCD bins are forced to [0-1200) after hz_to_cent conversion

#### morty v1.2.0
 - Corrected Jensen–Shannon distance and added Jeffrey's divergence
 - Converted intersection and cross-correlation to dissimilarities
 - Fixed the division by int problems by importing future division
 - Added merge method to PitchDistribution
 - Fixed the bin number mismatch in smoothen method in PitchDistribution
 - Changed minimum peak threshold
 - Refactored several variables and classes
 - Several bug fixes in unit conversion, distance matrix generation, ranking etc.

#### morty v1.1.0
 - Merged the classes Bozkurt and Chordia into a generic KNNClassifier class
 - Rewritten the input parsing, training and estimation methods in KNNClassifier
 - Created a separate KNN class for computing the nearest neighbors
 - Added 'min_peak_ratio' parameter to 'detect_peaks' method in the PitchDistribution class
 - Added Jensen–Shannon distance to KNN._distance
 - Refactored pitch extraction in extras and moved pitch slicing method there
 - Refactored all 'smooth_factor' parameters to 'kernel_width' for consistency
 - Removed save and load from PitchDistribution and created the methods to_json, from_json, to_pickle and from_pickle

#### morty v1.0.0
 - First stable release
 - Refactoring to improve readablity, maintanence and code quality

#### morty v0.9.0
 - First candidate release
 - Implemented [pitch and pitch-class distribution](https://github.com/altugkarakurt/morty/blob/master/morty/PitchDistribution.py).
 - Implemented methodologies proposed by [(A. C. Gedik and B.Bozkurt, 2010)](https://github.com/altugkarakurt/morty/blob/master/morty/Bozkurt.py) and [(P. Chordia and S. Şentürk, 2013)](https://github.com/altugkarakurt/morty/blob/master/morty/Chordia.py).
 - Implemented [mode and tonic estimation evaluation](https://github.com/altugkarakurt/morty/blob/master/morty/Evaluator.py) and [pitch unit conversion](https://github.com/altugkarakurt/morty/blob/master/morty/Converter.py).
 - Implemented experimental (basic) [multi-layer neural network](https://github.com/altugkarakurt/morty/blob/master/morty/NeuralNet.py) and [NN-based classifier](https://github.com/altugkarakurt/morty/blob/master/morty/NeuralClassifier.py).
