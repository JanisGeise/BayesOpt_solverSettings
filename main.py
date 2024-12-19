"""
handles everything
"""
import sys
import logging

from os.path import join
from os import environ, system

from OptimizeSolverSettings.execution import Executer
from OptimizeSolverSettings.optimize_settings import execute_optimization
from OptimizeSolverSettings.solver_options import GAMGSolverOptions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def set_openfoam_bashrc(training_path: str) -> None:
    # check if the path to bashrc was already added
    cmd = "source /usr/lib/openfoam/openfoam2206/etc/bashrc"

    with open(join(training_path, "Allrun"), "r") as f:
        check = [True for line in f.readlines() if line.startswith("# source bashrc")]

    # if not then add
    if not check:
        system(fr"sed -i '5i # source bashrc for OpenFOAM \n{cmd}' {training_path}/Allrun")
        system(fr"sed -i '5i # source bashrc for OpenFOAM \n{cmd}' {training_path}/Allrun.pre")


if __name__ == "__main__":
    # intervals to optimize, make sure the start times exist, and the end time of the last interval is <= the end time
    # of the base case
    interval = [(0, 0.5)]       # , (1, 1.5), (2, 2.5), (3, 3.5), (4, 4.5), (5, 5.5)]
    train_dir = "test_optimization"

    # set the path to openfoam
    environ["WM_PROJECT_DIR"] = "/usr/lib/openfoam/openfoam2206"
    sys.path.insert(0, environ["WM_PROJECT_DIR"])

    # instantiate executer
    # TODO: running in parallel increases elapsed time -> issue bc same problem for all optimization loops?
    executer = Executer(buffer_size=1, n_runner=1, run_directory=train_dir)

    # test: we need to add the path to bashrc of openfoam first if executed from within an IDE
    set_openfoam_bashrc(join(train_dir, "base"))

    # run the base case
    executer.prepare()

    # get all available parameters and corresponding options
    gamg_settings = GAMGSolverOptions()

    # create single param dict for each setting we want to test
    parameters = [
        {"name": "smoother", "type": "choice", "values": list(gamg_settings.keys["smoother"])},
        {"name": "interpolateCorrection", "type": "choice", "values": list(gamg_settings.keys["interpolateCorrection"])},
        {"name": "nFinestSweeps", "type": "range", "bounds": [1, 10]},
        {"name": "nPreSweeps", "type": "range", "bounds": [0, 10]},
        {"name": "nPostSweeps", "type": "range", "bounds": [0, 10]}
    ]

    # loop over intervals, optimize solver settings for each interval
    for idx, i in enumerate(interval):
        logger.info(f"Starting with optimization for interval {i}.")
        executer.start_time = i[0]
        executer.end_time = i[1]

        # execute the optimization loop
        execute_optimization(executer, parameters, i)
