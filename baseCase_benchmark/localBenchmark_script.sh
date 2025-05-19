#!/bin/bash
N=10
for i in {1..$N}; do
    echo "Starting simulation run $i"
    
    # Example: create a folder and copy base case
    mkdir -p ./openfoam_cases/run_$i
    cp -r base_sim/* ./openfoam_cases/run_$i/
    
    # Change to run dir and simulate
    cd ./openfoam_cases/run_$i
    bash Allrun.pre
    cd ../../

    echo "Finished simulation run $i"
done
