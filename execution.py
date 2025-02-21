"""Run parameter variations of base simulation to evaluate trial.
"""

from os import listdir
from os.path import isdir, join
from typing import Dict, Union
from copy import deepcopy
from collections import defaultdict
from smartsim import Experiment
from smartsim.entity import Model
import numpy as np
from pandas import read_csv


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


def run_parameter_variation(
    exp: Experiment, trials: dict, config: dict, time_idx: int
) -> Dict[int, float]:
    opt_config = config["optimization"]
    rs = exp.create_run_settings(exe="bash", exe_args="Allrun.solve")
    bs = None
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
            default[key_i] = val_i
        gamg_params[key] = default
    params_full = [sim_params | gamg_params[key] for key in gamg_params.keys()]
    params = defaultdict(list)
    for d in params_full:
        for key, val in d.items():
            params[key].append(val)
    keys_str = [str(key) for key in trials.keys()]
    ens = exp.create_ensemble(
        name=f"int_{time_idx}_trial_{'_'.join(keys_str)}",
        params=params,
        perm_strategy="step",
        run_settings=rs,
        batch_settings=bs,
    )
    base_case_path = config["simulation"]["base_case"]
    ens.attach_generator_files(to_configure=base_case_path)
    exp.generate(ens, overwrite=True, tag="!")
    exp.start(ens, block=True)
    return {
        key: extract_runtime(
            model,
            startTime,
            int(float(opt_config["duration"]) / float(opt_config["deltaT"])),
            opt_config["bad_value"],
        )
        for key, model in zip(trials.keys(), ens.models)
    }
