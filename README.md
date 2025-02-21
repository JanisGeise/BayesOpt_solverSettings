# Optimizing OpenFOAM's GAMG settings via Bayesian optimization

This project aims to use Bayesian optimization for finding the optimal solver settings in OpenFOAM. 
The basis of the project can be found in the repository 
[Learning of optimized solver settings for CFD applications](https://github.com/JanisGeise/learning_of_optimized_multigrid_solver_settings_for_CFD_applications)


## Getting started

### Dependencies

- OpenFOAM-v2406 or newer ([installation instructions]())
- Python 3.10 or newer

To set up a suitable virtual environment, run:
```
python3 -m venv bopt
source bopt/bin/activate
pip install -r requirements.txt
```

### Running an optimization

```
source bopt/bin/activate
python run.py example_config.yaml &> log.example_run
```

The script *eval_runs.py* contains a rudimentary example for visualizing the training outcome.

## Test case

The current test case is a 2D, laminar flow past a cylinder taken from the
[flow_data](https://github.com/AndreWeiner/flow_data) repository.

More test cases etc. will follow


## Still TODO

- early stopping
- SLURM configs for execution on HPC systems
- test different optimization configs for ax