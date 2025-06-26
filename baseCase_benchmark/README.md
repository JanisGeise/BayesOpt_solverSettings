# Benchmark for Base Case

This folder contains the files necesarry to carry out the benchmark to account for the uncertainty in execution time of the Openfoam simulation of Flow across cylinder. This is done to account for the changes in execution times within the same hardware.

## Simulation setup
The openfoam files for the required simulation to benchmark, must be placed in a folder called "base_sim". The Allrun should be named "Allrun.pre". This folder name and script name can be changed, but corresponding changes need to be made in the jobscript files.
## Benchmarking - Cluster - Parallel
The following steps are followed for benchmarking on the cluster, when it is possible to run benchmarks parallely:

1. The "clusterBenchmark_jobscript_parallel" file in the folder performs the same simulation for the required number of times. A few changes are to be made as required on the template jobscript file given.
2. Change the "CURR_DIR" to the appropriate directory of this folder.
3. No of simulations to be benchmarked can be changed by replacing the slurm directive "array=1-10", this command performs the simulation 10 times. Replace this to "array=1-12" to perform the benchmark 12 times.
4. Change other slurm directives as required
5. Submit the job using,
```bash
sbatch clusterBenchmark_jobscript_parallel
```
## Benchmarking - Cluster - Serial
The following steps are followed for benchmarking on the cluster, when benchmarks are to be run serially:

1. The "clusterBenchmark_jobscript_serial" file in the folder performs the same simulation for the required number of times. A few changes are to be made as required on the template jobscript file given.
2. Change the "CURR_DIR" to the appropriate directory of this folder.
3. No of simulations to be benchmarked can be changed by replacing the value of variable "N"
4. Change other slurm directives as required
5. Submit the job using,
```bash
sbatch clusterBenchmark_jobscript_serial
```

## Benchmarking - Local
The following steps are followed for benchmarking on the local machine:

1. Use the "localBenchmark_script.sh" file to run the benchmark on the local machine.
2. Change the number of times to benchmark by changing the variable "N" in the script.
3. Run the script using
```bash
bash localBenchmark_script.sh
```
## Data generation
The execution time data of the benchmark cases is required for the post processing of BO run, which is done using the "eval_runs.py" script. To extract this data in the required structure, use the "copy_data.sh" script. A new folder called "benchmark_time_info" is created, where the required data is stored. 
```bash
bash copy_data.sh
```
## Plotting
The plotter uses the data in "benchmark_time_info" to create plots. The total simulation time and time step needs to be provided as arguments for the plotting. To visualize this data, an example usage of "plot.py" file is 
```bash
python3 plot.py --duration 120.0 --deltaT 0.05
```

