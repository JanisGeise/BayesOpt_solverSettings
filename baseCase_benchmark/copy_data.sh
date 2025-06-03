#!/bin/bash

# Create a destination folder if it doesn't exist
mkdir -p benchmark_time_info

# Loop over run_1 to run_10
for i in {1..10}; do
    src_dir="openfoam_cases/run_$i/postProcessing/time/0/timeInfo.dat"
    dest_dir="benchmark_time_info/timeInfo_$i.dat"
    cp -r "$src_dir" "$dest_dir"
done
