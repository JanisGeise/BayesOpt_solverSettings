# Bayesian optimization for finding optimal solver settings in OpenFOAM
This project aims to use Bayesian optimization for finding the optimal solver settings in OpenFOAM. 
The basis of the project can be found in the repository 
[Learning of optimized solver settings for CFD applications](https://github.com/JanisGeise/learning_of_optimized_multigrid_solver_settings_for_CFD_applications)

Currently, [Ax](https://ax.dev/) is utilized, which requires Python version $\ge 3.10$.

## Test case
The current test case is a 2D, laminar flow past a cylinder taken from the
[flow_data](https://github.com/AndreWeiner/flow_data) repository.

More test cases etc. will follow


## Still TODO
- fix saving and loading of optimization results
- fix plotting scripts of optimization history
- fix restart option
- deactivate cuda usage / warning msg
- test on medium grid
- check used policies for optimization
- early stopping
- parallel execution of simulations
- SLURM configs for execution on HPC systems
- test different optimization configs for ax
- extend for other solver types (not only GAMG) -> implement new solver settings classes
- ...