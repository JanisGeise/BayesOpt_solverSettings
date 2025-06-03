"""Run parameter variations of base simulation to evaluate trial.
"""

from os import listdir
from os.path import isdir, join
from typing import Dict, Union
from copy import deepcopy
from collections import defaultdict
from math import sqrt
from smartsim import Experiment
from smartsim.settings import RunSettings
from smartsim.settings.base import BatchSettings
from smartsim.entity import Model
import numpy as np
from pandas import read_csv


LOG10_KEYS = ("tolerance", "relTol", "toleranceCoarsest", "relTolCoarsest")


def is_float(value: str) -> bool:
    try:
        _ = float(value)
        return True
    except:
        return False


def find_closest_time(path: str, time: Union[float, int]) -> str:
    dirs = [f for f in listdir(path) if isdir(join(path, f)) and is_float(f)]
    dirs_num = np.array([float(f) for f in dirs])
    closest = np.argmin(np.absolute(dirs_num - time))
    return dirs[closest]


def extract_runtime(
    model: Model, startTime: str, steps: int, bad_value: float
) -> float:
    path = join(model.path, "postProcessing", "time", startTime, "timeInfo.dat")
    try:
        df = read_csv(
            path,
            header=None,
            sep=r"\s+",
            skiprows=1,
            usecols=[0, 1],
            names=["t", "t_cpu_cum"],
        )
        if len(df.t) < int(0.95 * steps):
            return bad_value
        else:
            # discard first time step
            # normalize with the number of time steps
            t_cum = df["t_cpu_cum"].values
            return (t_cum[-1] - t_cum[0]) / (len(t_cum) - 1)
    except:
        return bad_value
    

def batch_settings_from_config(exp: Experiment, batch_config: dict) -> Union[BatchSettings, None]:
    if batch_config is not None:
        bs = exp.create_batch_settings(batch_args=batch_config.get("batch_args"))
        if "preamble" in batch_config:
            bs.add_preamble(batch_config["preamble"])
    else:
        bs = None
    return bs


def run_parameter_variation(
    exp: Experiment, trials: dict, config: dict, time_idx: int
) -> Dict[int, float]:
    opt_config = config["optimization"]
    rs = RunSettings(exe="bash", exe_args="Allrun.solve")
    bs = batch_settings_from_config(exp, config.get("batch_settings"))
    path = join(exp.exp_path, "base_sim", "processor0")
    startTime = find_closest_time(path, opt_config["startTime"][time_idx])
    endTime = float(startTime) + opt_config["duration"]
    sim_params = {
        "startTime": float(startTime),
        "endTime": endTime,
        "writeInterval": opt_config["writeInterval"],
        "deltaT" : opt_config["deltaT"],
        "baseCase": "../../base_sim",
    }
    gamg_params = {}
    for key in trials.keys():
        default = deepcopy(config["simulation"]["gamg"])
        for key_i, val_i in trials[key].items():
            if key_i in LOG10_KEYS:
                default[key_i] = 10**val_i
            else:
                default[key_i] = val_i
        gamg_params[key] = default
    params_full = [sim_params | gamg_params[key] for key in gamg_params.keys()]
    params = defaultdict(list)
    for d in params_full:
        for key, val in d.items():
            params[key].extend([val] * opt_config["n_repeat_trials"])
    keys_str = [str(key) for key in trials.keys()]
    ens = exp.create_ensemble(
        name=f"int_{time_idx}_trial_{'_'.join(keys_str)}",
        params=params,
        perm_strategy="step",
        run_settings=rs,
        batch_settings=bs
    )
    base_case_path = config["simulation"]["base_case"]
    ens.attach_generator_files(to_configure=base_case_path)
    exp.generate(ens, overwrite=True, tag="!")
    if opt_config["repeated_trials_parallel"]:
        exp.start(ens, block=True)
    else:
        n_parallel = opt_config["batch_size"]
        for i in range(0, len(ens.models), n_parallel):
            ens_batch = ens.models[i:i+n_parallel]
            for model_i in ens_batch:
                model_i.batch_settings = bs
            exp.start(*ens_batch, block=True)
    runtimes = [
        extract_runtime(
            model,
            startTime,
            int(float(opt_config["duration"]) / float(opt_config["deltaT"])),
            opt_config["bad_value"],
        )
        for model in ens.models
    ]
    nr = opt_config["n_repeat_trials"]
    if nr > 1:
        stats = [
            (np.mean(runtimes[i:i+nr]), np.std(runtimes[i:i+nr]) / sqrt(nr))
            for i in range(0, len(runtimes), nr)
        ]
        obj = {
            key : stat_i for key, stat_i in zip(trials.keys(), stats)
        }
    else:
        obj = {
            key : (t_i, 0.0) for key, t_i in zip(trials.keys(), runtimes)
        }
    return obj
