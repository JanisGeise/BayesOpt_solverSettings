"""
    executes the Bayesian optimization loop
"""
import logging

from glob import glob
from torch import device
from os.path import join
from copy import deepcopy
from pandas import read_csv
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.early_stopping.strategies import PercentileEarlyStoppingStrategy

from .execution import Executer
from .solver_options import GAMGSolverOptions
from .manipulate_settings import ManipulateSolverSettings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OptimizeSolverSettings:
    def __init__(self, training_path: str, buffer_size: int, n_runner: int = 1, simulation: str = "cylinder_2D_Re100",
                 seed: int = 0):
        self._buffer_size = buffer_size
        self._executer = Executer(simulation=simulation, buffer_size=buffer_size, n_runner=n_runner,
                                  run_directory=training_path)
        self._ax = AxClient(random_seed=seed, torch_device=device("cpu"))
        self._parser = ManipulateSolverSettings(self._executer.run_directory)
        self._gamg_settings = GAMGSolverOptions()
        self._solver_dict = self._gamg_settings.create_default_dict()
        self._dt_base = None

    def prepare(self):
        self._executer.prepare()

        # determine execution time for the baseline case
        self._dt_base = self._compute_elapsed_time(join(self._executer.run_directory, "base"))

    def optimize(self, intervals: list[tuple], parameters: list[dict], n_trials_max: int = 50,
                 restart: bool = False) -> None:
        # we assume that the parameters to optimize remain the same for all intervals
        for idx, i in enumerate(intervals):
            logger.info(f"Starting with optimization for interval {i}.")
            self._executer.start_time = i[0]
            self._executer.end_time = i[1]

            # execute the optimization loop
            self._execute_optimization(parameters, i, n_trials_max, restart)

    def _compute_elapsed_time(self, load_path: str) -> float:
        """
        compute the elapsed CPU time for a given interval

        :param load_path: path to the directory should be either 'base' or 'copy_*'
        :return: elapsed CPU time between the start and the end of the interval
        """
        t = read_csv(glob(join(load_path, "postProcessing", "time", "*", "timeInfo.dat"))[0], header=None,
                     sep=r"\s+", skiprows=1, usecols=[0, 1], names=["t", "t_cpu_cum"])

        # execution time is logged beginning after the first time step, except for the base case it will always be zero
        start = t["t_cpu_cum"][t["t"] == self._executer.start_time].values
        if start.size == 0:
            start = 0

        # if simulation didn't converge assign high value, else assign the correct elapsed time
        if t["t"].iloc[-1] < self._executer.end_time:
            t_end = 9999
        else:
            t_end = t["t_cpu_cum"][t["t"] == self._executer.end_time].values
        return float(t_end - start)

    def _run_trial(self, solver_dict: list) -> dict:
        """
        executes a single optimization for a series of solver settings

        :param solver_dict: dict containing the solver settings to run the simulation with
        :return: dict containing the loss as a ratio between execution time with the current settings and the baseline
        """
        # update dict for all copies; we need to make a deepcopy, because the solver dicts are reset when updated
        for no in range(self._buffer_size):
            self._parser.replace_settings(deepcopy(solver_dict[no]), f"copy_{no}")

        # execute the simulations in the copy_* directories with the new settings
        self._executer.execute()

        # get the execution time for each setting and compare it to the baseline
        dt_copies = [self._compute_elapsed_time(join(self._executer.run_directory, f"copy_{i}")) for i in
                     range(self._buffer_size)]

        return {"loss": tuple([i / self._dt_base for i in dt_copies])}

    def _execute_optimization(self, parameters: list[dict], i: tuple, n_trials_max: int = 50,
                              restart: bool = False) -> None:
        """
        wrapper for executing the Bayesian optimization loop for finding the optimal solver settings for a given
        interval

        :param parameters: list containing the dicts for the solver settings to optimize
        :param i: current interval
        :param n_trials_max: max. number of optimization iterations
        :param restart: a restart option for continue a previous optimization
        :return: None
        """
        # load if option restart is set TODO: model still starts from scratch if restart is set
        if restart:
            raise NotImplementedError
            # logger.info("Loading state dict.")
            # throws error
            # ax.load_experiment(join(executer.run_directory, f"client_interval_{i[0]}_to_{i[1]}.json"))
            # ax.load_from_json_file(join(executer.run_directory, f"client_interval_{i[0]}_to_{i[1]}.json"))

        else:
            # test early stopping, only sensible when buffer_size > 2
            # stopping = PercentileEarlyStoppingStrategy(percentile_threshold=70, min_progression=0.3, min_curves=2,
            #                                            trial_indices_to_ignore=[0], seconds_between_polls=1,
            #                                            normalize_progressions=True)

            self._ax.create_experiment(name="experiment", parameters=parameters, overwrite_existing_experiment=True,
                                       objectives={"loss": ObjectiveProperties(minimize=True)})

        # execute the optimization loop
        """
        we get N_buffer results for each setting we test -> Ax determines the uncertainty for executing the same 
        parameters multiple times. So not sure how we can test multiple parameters in parallel...
        """
        for _ in range(n_trials_max):
            next_parameters, trial_index = self._ax.get_next_trial()

            # replace solver dict settings with new settings (currently the same settings for all copies, see comment
            # above)
            all_solver_dicts = []
            for _ in range(self._buffer_size):
                for key, value in next_parameters.items():
                    self._solver_dict[key] = value
                all_solver_dicts.append(self._solver_dict)

            loss = self._run_trial(all_solver_dicts)
            if self._buffer_size > 1:
                self._ax.complete_trial(trial_index=trial_index, raw_data=loss)
            else:
                self._ax.complete_trial(trial_index=trial_index, raw_data={"loss": loss["loss"][0]})

            # save optimization results for each interval, save() raises NotImplementedError
            self._ax.save_to_json_file(join(self._executer.run_directory, f"client_interval_{i[0]}_to_{i[1]}.json"))

        # print some output
        logger.info(f"\nOptimal parameters for interval {i}: {self._ax.get_best_parameters()} \n")


if __name__ == "__main__":
    pass
