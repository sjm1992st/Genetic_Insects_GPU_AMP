========================================================================
    CONSOLE APPLICATION : GA_parallel_Amp Project Overview
========================================================================

This project is an attempt to use the GPU to "accelerate" the process of
genetic algorithms. So far the results have not shown a significant speed
improvement however the project was fun to create because the subject of 
GA has a diverse, dynamic and interesting set of algorithms.

I found a real problem with the roulette wheel implementation. I selected
to process a matrix of dimension (population_size ^2) with row and columns
storing indexes into the population for parent selection. This is based
on the CDF cumulative distribution function of the fitness of the population
rescaled to be between 0-1. The problem was that I had to lookup the CDF index
based on an input value between 0 and 1, so taking these and scaling
to discrete intervals of 1/population_size, then finding the Row value for 
each column ... this was then indexed using the random values for parent selection
to compute the inverse CDF (although I may have gone wrong somewhere - there is 
an implementation of this in the main function).

The process then finds the index with a parallel prefix sum scan of the Rows.
However this method did not work in practice and so for choosing parents the working implementation still uses
the vanilla GA method.