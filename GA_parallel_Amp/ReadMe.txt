========================================================================
    CONSOLE APPLICATION : GA_parallel_Amp Project Overview
========================================================================

This project is an attempt to use the GPU to "accelerate" the process of
genetic algorithms. So far the results have not shown a significant speed
improvement however the project was fun to create because the subject of 
GA has a diverse, dynamic and interesting set of algorithms.

I found a real problem with the roulette wheel implementation - the sample computes
the cumulative distribution function of the fitness, then rescales to the range [0-1].
The input random numbers then have to find the index of the CDF array that corresponds 
to their value - this is done using a matrix of threads of size (population^2). The
inverse CDF gives the index, the index is stored in the matrix, the rest of the entries 
are zero, then a parallel prefix sum scan is performed over each row to find the 
index, these indexes are then stored in a vector.

The method is much much slower than the C++ version, however with more optimization
and large populations and gene sizes (the genes are compressed into bit format in 
a single 32 bit integer) the optimizations should start to show that the parallel
version is faster.

This version completed for me in about 133 epochs, it is best compiled in release mode.