### Dynamic Time Warping Motif Discovery
Python implementation of [Matrix Profile XXII: Exact Discovery of Time Series Motifs under DTW - Sara Alaee, Kaveh Kamgar, Eamonn Keogh](https://arxiv.org/pdf/2009.07907.pdf)

Adapted from the [MATLAB implementation](https://sites.google.com/site/dtwmotifdiscovery/)

#### Usage
Those functions written in C++ are copied directly from the MATLAB implementation. They need to be compiled to shared object files before use, these compiled library files are provided (.so) and can be reproduced using the following

```g++ -std=c++11 -fPIC -shared -o dtw_upd.so dtw_upd.cpp```