import sys
from os import makedirs
from os.path import join
from yaml import safe_load
from pandas import read_csv
import numpy as np
from smartsim import Experiment
from smartsim.settings import RunSettings, MpirunSettings, SrunSettings
from execution import batch_settings_from_config


# load settings
config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
try:
    with open(config_file, "r") as cf:
        config = safe_load(cf)
except Exception as e:
    print(e)

# set up execution infrastructure
makedirs(config["experiment"]["exp_path"], exist_ok=True)
exp = Experiment(**config["experiment"])

# execute Allrun.pre
sim_config = config["simulation"]
base_case_path = sim_config["base_case"]
base_name = base_case_path.split("/")[-1]
rs = RunSettings(exe="bash", exe_args="Allrun.pre")
bs = batch_settings_from_config(exp, config.get("batch_settings"))
params = {
    "startTime" : sim_config["startTime"],
    "endTime": sim_config["startTime"] + sim_config["duration"],
    "writeInterval" : sim_config["writeInterval"],
    "deltaT" : sim_config["deltaT"]
} | sim_config["gamg"]
base_sim = exp.create_model(
    "base_sim",
    params=params,
    run_settings=rs,
    batch_settings=bs,
)
base_sim.attach_generator_files(to_configure=base_case_path)
exp.generate(base_sim, overwrite=True, tag="!")
exp.start(base_sim, block=True, summary=True)

# create setups for repeated runs
block = not config["optimization"]["repeated_trials_parallel"]
n_repeat = config["optimization"]["n_repeat_trials"]
rs = RunSettings(exe="bash", exe_args="link_procs")
params["baseCase"] = "../base_sim"
models_repeat = []
for i in range(n_repeat):
    models_repeat.append(
        exp.create_model(
            f"base_sim_{i}",
            params=params,
            run_settings=rs,
            batch_settings=None
        )
    )
    models_repeat[-1].attach_generator_files(to_configure=base_case_path)
    exp.generate(models_repeat[-1], overwrite=True, tag="!")
    if i == n_repeat - 1:
        block = True
    exp.start(models_repeat[-1], block=block, summary=True)

# run solver for each copy
block = not config["optimization"]["repeated_trials_parallel"]
launcher = config["experiment"]["launcher"]
settings_class = MpirunSettings if launcher == "local" else SrunSettings
solver = sim_config["solver"]
for i in range(n_repeat):
    solver_settings = settings_class(
        exe=solver,
        exe_args=f"-case {models_repeat[i].path} -parallel",
        run_args=sim_config["run_args"]
    )
    solver_model = exp.create_model(
        name=f"{solver}_{i}",
        run_settings=solver_settings,
        batch_settings=bs
    )
    if i == n_repeat - 1:
        block = True
    exp.start(solver_model, block=block, summary=True)

# collect and save timings
block_timing = []
for i in range(n_repeat):
    path = join(exp.exp_path, f"base_sim_{i}", "postProcessing", "time", "0", "timeInfo.dat")
    df = read_csv(
            path,
            header=None,
            sep=r"\s+",
            skiprows=1,
            usecols=[0, 1],
            names=["t", "t_cpu_sum"],
        )
    block_timing.append(df.t_cpu_sum.values.reshape((-1, 1)))
block_timing = np.concatenate(block_timing, axis=1)
np.save(join(exp.exp_path, "cum_cpu_time.npy"), block_timing)