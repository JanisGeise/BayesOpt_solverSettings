import sys
from os import makedirs
from os.path import join
from yaml import safe_load
from pandas import read_csv
import numpy as np
from smartsim import Experiment
from smartsim.settings import RunSettings
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

# run base case if necessary
sim_config = config["simulation"]
base_case_path = sim_config["base_case"]
base_name = base_case_path.split("/")[-1]
rs = RunSettings(exe="bash", exe_args="Allrun.pre")
bs = batch_settings_from_config(exp, config.get("batch_settings"))
block = not config["optimization"]["repeated_trials_parallel"]
n_repeat = config["optimization"]["n_repeat_trials"]
#for i in range(n_repeat):
#    base_sim = exp.create_model(
#        f"base_sim_{i}",
#        params={
#                "startTime" : sim_config["startTime"],
#                "endTime": sim_config["startTime"] + sim_config["duration"],
#                "writeInterval" : sim_config["writeInterval"],
#                "deltaT" : sim_config["deltaT"]
#            } | sim_config["gamg"],
#        run_settings=rs,
#        batch_settings=bs,
#    )
#    base_sim.attach_generator_files(to_configure=base_case_path)
#    exp.generate(base_sim, overwrite=True, tag="!")
#    if i == n_repeat - 1:
#        block = True
#    exp.start(base_sim, block=block, summary=True)

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

    