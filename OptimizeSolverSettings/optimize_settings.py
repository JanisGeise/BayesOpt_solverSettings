"""
    executes the Bayesian optimization loop
"""
import logging

from glob import glob
from typing import Union
from torch import device
from os.path import join
from pandas import read_csv
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

from .execution import Executer
from .solver_options import GAMGSolverOptions
from .manipulate_settings import ManipulateSolverSettings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def compute_elapsed_time(load_path: str, t_start: Union[int, float], t_end: Union[int, float]) -> float:
    """
    compute the elapsed CPU time for a given interval

    :param load_path: path to the directory should be either 'base' or 'copy_*'
    :param t_start: interval's physical start time
    :param t_end: interval's physical end time
    :return: elapsed CPU time between the start and the end of the interval
    """
    t = read_csv(glob(join(load_path, "postProcessing", "time", "*", "timeInfo.dat"))[0], header=None,
                 sep=r"\s+", skiprows=1, usecols=[0, 1], names=["t", "t_cpu_cum"])

    # execution time is logged beginning after the first time step, so except for the base case it will always be zero
    start = t["t_cpu_cum"][t["t"] == t_start].values
    if start.size == 0:
        start = 0

    # if simulation didn't converge assign high value, else assign the correct elapsed time
    t_end = 9999 if t["t"].iloc[-1] < t_end else t["t_cpu_cum"][t["t"] == t_end].values
    return float(t_end - start)


def run_trial(executer: Executer, solver_dict: list) -> dict:
    """
    executes a single optimization for a series of solver settings

    :param executer: executer handling the parallel execution of the simulations
    :param solver_dict: dict containing the solver settings to run the simulation with
    :return: dict containing the loss as a ratio between execution time with the current settings and the baseline
    """
    # determine execution time for the baseline case
    t_start, t_end = executer.start_time, executer.end_time
    dt_base = compute_elapsed_time(join(executer.run_directory, "base"), t_start, t_end)

    # instantiate parser for manipulating the solver settings, assume all copies have the same settings
    parser = ManipulateSolverSettings(executer.run_directory)

    # update dict for all copies
    for no in range(executer.buffer_size):
        parser.replace_settings(solver_dict[no], f"copy_{no}")

    # execute the simulations in the copy_* directories with the new settings
    executer.execute()

    # get the execution time for each setting and compare it to the baseline
    dt_copies = [compute_elapsed_time(join(executer.run_directory, f"copy_{i}"), t_start, t_end) for i in
                 range(executer.buffer_size)]

    return {"loss": tuple([i / dt_base for i in dt_copies])}


def execute_optimization(executer: Executer, parameters: list[dict], i: tuple, seed: int = 0, n_trials_max: int = 15,
                         restart: bool = False):
    """
    wrapper for executing the Bayesian optimization loop for finding the optimal solver settings for a given interval

    :param executer: executer handling the parallel execution of the simulation
    :param parameters: list containing the dicts for the solver settings to optimize
    :param i: current interval
    :param seed: seed value
    :param n_trials_max: max. number of optimization iterations
    :param restart: a restart option for continue a previous optimization
    :return: None
    """
    # don't run on GPU
    ax = AxClient(random_seed=seed, torch_device=device("cpu"))

    # load if option restart is set TODO: model still starts from scratch if restart is set
    if restart:
        raise NotImplementedError
        # logger.info("Loading state dict.")
        # throws error
        # ax.load_experiment(join(executer.run_directory, f"client_interval_{i[0]}_to_{i[1]}.json"))
        # ax.load_from_json_file(join(executer.run_directory, f"client_interval_{i[0]}_to_{i[1]}.json"))

    else:
        # TODO: how determine convergence? -> early_stopping as kwarg -> usage?
        ax.create_experiment(name="experiment", parameters=parameters,
                             overwrite_existing_experiment=True if restart else False,
                             objectives={"loss": ObjectiveProperties(minimize=True)})

    # initialize solver dict, get all mandatory settings to make sure nothing is missing
    gamg_settings = GAMGSolverOptions()
    solver_dict = gamg_settings.create_default_dict()

    # execute the optimization loop
    for _ in range(n_trials_max):
        # TODO: how to sample N_buffer different settings for each runner? -> we need to create N_buffer dicts
        next_parameters, trial_index = ax.get_next_trial()

        # replace solver dict settings with new settings
        all_solver_dicts = []
        for _ in range(executer.buffer_size):
            for key, value in next_parameters.items():
                solver_dict[key] = value
            all_solver_dicts.append(solver_dict)

        loss = run_trial(executer, all_solver_dicts)
        # TODO: sequential doesn't work with tuple of len 1, so unpack
        ax.complete_trial(trial_index=trial_index, raw_data={"loss": loss["loss"][0]})

        # save optimization results for each interval, save() raises NotImplementedError
        ax.save_to_json_file(join(executer.run_directory, f"client_interval_{i[0]}_to_{i[1]}.json"))

    # print some output
    logger.info(f"\nOptimal parameters for interval {i}:", ax.get_best_parameters(), "\n")


if __name__ == "__main__":
    pass
