# Benchmark for Base Case

This folder contains the files necesarry to carry out the benchmark to account for the uncertainty in execution time of the Openfoam simulation of Flow across cylinder. This is done to account for the changes in execution times within the same hardware.

## Benchmarking - Cluster
The following steps are followed for benchmarking on the cluster:

1. The "clusterBenchmark_jobscript" file in the folder performs the same simulation for the required number of times. A few changes are to be made as required on the template jobscript file given.
2. Change the "CURR_DIR" to the appropriate directory of this folder.
3. No of simulations to be benchmarked can be changed by replacing the slurm directive "array=1-10", this command performs the simulation 10 times. Replace this to "array=1-12" to perform the benchmark 12 times.
4. Change other slurm directives as required
5. Submit the job using,
```bash
sbatch clusterBenchmark_jobscript
```

## Benchmarking - Local
The following steps are followed for benchmarking on the local machine:

1. Use the "localBenchmark_script.sh" file to run the benchmark on the local machine.
2. Change the number of times to benchmark by changing the variable "N" in the script.
3. Run the script using
```bash
bash localBenchmark_script.sh
```
## Plotting
The simulation files are stored in a folder called "openfoam_cases". To Visulize this data use the "plot.py" file. 
```bash
python3 plot.py
```
