# PostProcessing of BO trials

This folder contains the files necesarry to carry out the post-processing of the BO trials. 4 types of plots are output from the code

1. "trial vs base" - It compares the execution time of the trials at different intervals with the base case
2. "best parms" - Makes a parallel plot of best parameters from the optimization across different intervals
3. "trial vs obj" - Plot to check how the objective function varies over trials
4. "gaussian prcoess" - Makes the gaussian process plot for a given parameter. Currently works if only one parameter is being optimized, comment this out if multiple parameters are being optimized.
