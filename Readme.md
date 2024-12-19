# BayesOpt solver settings
idea is ti use Bayesian Optimization for optimizing multigrid solver settings in OpenFoam as it is done
in [Learning of optimized solver settings for CFD applications]([https://github.com/JanisGeise/learning_of_optimized_multigrid_solver_settings_for_CFD_applications])


### Test case
cylinder2D from [flow_data](https://github.com/AndreWeiner/flow_data) repository

Ax requires Python $\ge 3.10$

the rest will be coming soon...

### TODO
- fix saving and loading of optimization results
- fix plotting scripts
- fix restart option
- deactivate cuda usage / warning msg
- test on medium grid
- check used policies for optimization
- early stopping
- parallel execution of simulations
- refactoring to OOP
- SLURM configs for execution on HPC systems
- test different optimization configs for ax
- extend for other solver types (not only GAMG)
- ...