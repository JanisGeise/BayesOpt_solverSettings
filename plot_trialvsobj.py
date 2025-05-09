import sys
from os.path import join
from yaml import safe_load
from ax.service.ax_client import AxClient
from ax.modelbridge import Models
from ax.plot.contour import interact_contour_plotly, interact_contour
from ax.utils.report.render import render_report_elements
from ax.plot.render import plot_config_to_html
from ax.plot.render import plotly_offline
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv

plt.style.use('dark_background')
plt.rcParams.update({
    "text.usetex": False
})

# load settings
config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
try:
    with open(config_file, "r") as cf:
        config = safe_load(cf)
except Exception as e:
    print(e)

data_li = []
for i, st_i in enumerate(config["optimization"]["startTime"][:]):
    ax_client = AxClient().load_from_json_file(join(f"{config['experiment']['exp_path']}", f"ax_client_int_{i}.json"))

    data = ax_client.experiment.fetch_data().df["mean"].values
    df = ax_client.experiment.fetch_data().df
    df["best"] = df["mean"].cummin()
    data_li.append(df)

for i, st_i in enumerate(config["optimization"]["startTime"][:]):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.scatter(data_li[i]["trial_index"], data_li[i]["mean"], marker="x", label="trial")
    ax.plot(data_li[i]["trial_index"], data_li[i]["best"], label="best param")
    ax.legend()
    ax.set_xlabel("trial index")
    ax.set_ylabel("obj")
    ax.set_title("objective v/s trials, interval={}".format(st_i))
    fig.savefig("trialvsobj_int_{}.svg".format(st_i), bbox_inches="tight", transparent=True)

