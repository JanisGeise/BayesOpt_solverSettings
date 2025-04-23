import sys
from os import makedirs
from os.path import isdir, join
from itertools import islice
from logging import INFO
from yaml import safe_load
import pandas as pd
from smartsim import Experiment
from execution import run_parameter_variation, batch_settings_from_config


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
rs = exp.create_run_settings(exe="bash", exe_args="Allrun.pre")
bs = batch_settings_from_config(exp, config.get("batch_settings"))
if not isdir(join(exp.exp_path, "base_sim")):
    base_sim = exp.create_model(
        "base_sim",
        params={
                "startTime" : sim_config["startTime"],
                "endTime": sim_config["startTime"] + sim_config["duration"],
                "writeInterval" : sim_config["writeInterval"],
                "deltaT" : sim_config["deltaT"]
            } | sim_config["gamg"],
        run_settings=rs,
        batch_settings=bs,
    )
    base_sim.attach_generator_files(to_configure=base_case_path)
    exp.generate(base_sim, overwrite=True, tag="!")
    exp.start(base_sim, block=True, summary=True)

# perform repeated runs
def chunk_dict(d, chunk_size):
    it = iter(d.items())
    while True:
        chunk = dict(islice(it, chunk_size))
        if not chunk:
            break
        yield chunk

opt_config = config["optimization"]
for i, startTime in enumerate(opt_config["startTime"]):
    trials = {j : sim_config["gamg"] for j in range(opt_config["repeat_trials"])}
    results = {}
    for chunk in chunk_dict(trials, opt_config["batch_size"]):
        results = results | run_parameter_variation(exp=exp, trials=chunk, config=config, time_idx=i)
    print(results, results.values())
    pd.DataFrame(results.values(), columns=("time",)).to_csv(join(exp.exp_path, f"timing_int_{i}.csv"))