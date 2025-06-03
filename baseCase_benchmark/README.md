# Benchmark for Base Case

This folder contains the files necesarry to carry out the benchmark to account for the uncertainty in execution time of the Openfoam simulation of Flow across cylinder. This is done to account for the changes in execution times within the same hardware.

## Simulation setup
The openfoam files for the required simulation to benchmark, must be placed in a folder called "base_sim". The Allrun should be named "Allrun.pre". This folder name and script name can be changed, but corresponding changes need to be made in the jobscript files.
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
The simulation files are stored in a folder called "openfoam_cases". To Visualize this data use the "plot.py" file. 
```bash
python3 plot.py
```
## Data generation
The execution time data of the benchmark cases is required for the post processing of BO run, which is done using the "eval_runs.py" script. To extract this data in the required structure, use the "copy_data.sh" script. A new folder called "benchmark_time_info" is created, where the required data is stored. 
```bash
bash copy_data.sh
```
