import sys
from os import makedirs
from os.path import isdir, join
from math import log10
from logging import INFO
from yaml import safe_load
from smartsim import Experiment
from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
from ax.service.utils.instantiation import ObjectiveProperties
from ax.global_stopping.strategies.improvement import ImprovementGlobalStoppingStrategy
from ax.utils.common.logger import get_logger
from ax.storage.json_store.save import save_experiment
from execution import run_parameter_variation, batch_settings_from_config, LOG10_KEYS


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

# perform optimization
opt_config = config["optimization"]
logger = get_logger(name="ax")
logger.setLevel(INFO)
for i, startTime in enumerate(opt_config["startTime"]):
    gs = GenerationStrategy(
        steps=[
            GenerationStep(
                model=Models.SOBOL,
                num_trials=opt_config["sobol_trials"],
                max_parallelism=opt_config["sobol_trials"],
            ),
            GenerationStep(
                model=Models.BO_MIXED,
                num_trials=opt_config["bo_trials"],
                max_parallelism=opt_config["batch_size"],
            ),
        ]
    )
    stopping_strategy = ImprovementGlobalStoppingStrategy(**opt_config["stopping"])
    ax_client = AxClient(
        random_seed=opt_config["seed"],
        generation_strategy=gs,
        global_stopping_strategy=stopping_strategy
    )
    ax_client.create_experiment(
        name=f"{config['experiment']['name']}-ax-{i}",
        parameters=list(opt_config["gamg"].values()),
        overwrite_existing_experiment=True,
        objectives={"execution_time": ObjectiveProperties(minimize=True)},
    )

    complete_bo = False
    complete_sobol = False
    default_trial = {
        key : (log10(float(sim_config["gamg"][key])) if key in LOG10_KEYS else sim_config["gamg"][key])
        for key in opt_config["gamg"].keys()
    }
    trial, idx = ax_client.attach_trial(default_trial)
    idx, obj = list(run_parameter_variation(exp=exp, trials={idx : trial}, config=config, time_idx=i).items())[0]
    ax_client.complete_trial(trial_index=idx, raw_data={"execution_time" : obj})
    while not (complete_sobol and complete_bo):
        trials, complete = ax_client.get_next_trials(opt_config["batch_size"])
        if not trials:
            complete, complete_sobol, complete_bo = True, True, True
        if complete and not complete_sobol:
            complete_sobol = True
            complete = False
            complete_bo = complete and complete_sobol
        if not complete:
            logger.info("running parameter variation again")
            results = run_parameter_variation(exp=exp, trials=trials, config=config, time_idx=i)
            for idx, obj in results.items():
                ax_client.complete_trial(trial_index=idx, raw_data={"execution_time" : obj})
        else:
            logger.info("All trials complete. Saving results.")
            ax_client.save_to_json_file(join(exp.exp_path, f"ax_client_int_{i}.json"))
            save_experiment(experiment=ax_client.experiment, filepath=join(exp.exp_path, f"ax_experiment_int_{i}.json"))