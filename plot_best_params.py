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
import pandas as pd

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

dic_best = {}
for i, st_i in enumerate(config["optimization"]["startTime"][:]):
    ax_client = AxClient().load_from_json_file(join(f"{config['experiment']['exp_path']}", f"ax_client_int_{i}.json"))
    dic_best[float(st_i)] = ax_client.get_best_parameters()


parallel_data = []
for x in dic_best.keys():
    parallel_dic = dic_best[x][0]
    parallel_dic["interval"] = x
    parallel_data.append(parallel_dic)
    
df = pd.DataFrame(parallel_data)

li = list(dic_best[float(st_i)][0].keys())
li.remove("interval")
plot_li = ['interval']
plot_li.extend(li)
plot_df = df[plot_li]

# Plot
for xi in li:
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(plot_df["interval"], plot_df[xi], marker="o")
    ax1.set_title("Best parameter - {} - for different intervals".format(xi))
    ax1.set_xlabel("interval")
    ax1.set_ylabel(xi)
    fig1.savefig("parallel_plot_best_params_{}.svg".format(xi), bbox_inches="tight", transparent=True)
