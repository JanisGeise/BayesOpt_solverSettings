#!/bin/bash

#SBATCH --output=base.out
#SBATCH --error=base.err
#SBATCH --job-name=base_sim
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --ntasks=2
#SBATCH --partition=barnard
module load release/24.04 GCC/12.3.0 OpenMPI/4.1.5 OpenFOAM/v2406
source $FOAM_BASH

cd /data/horse/ws/tata993f-general_testing/base_case_test/base_sim

bash Allrun.pre 
