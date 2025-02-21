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
    "text.usetex": True
})

# load settings
config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
try:
    with open(config_file, "r") as cf:
        config = safe_load(cf)
except Exception as e:
    print(e)

base_sim_path = join(f"{config['experiment']['exp_path']}", "base_sim")
base_timing = read_csv(
    join(base_sim_path, "postProcessing", "time", "0", "timeInfo.dat"),
    header=None,
    sep=r"\s+",
    skiprows=1,
    usecols=[0, 1],
    names=["t", "t_cpu_cum"],
)
t_plot = base_timing.t.values[49::50] - 0.05
t_cum_plot = base_timing.t_cpu_cum.values[49::50]
t_cum_plot = np.concatenate((t_cum_plot[:1], t_cum_plot[1:]-t_cum_plot[:-1]))
fig, ax = plt.subplots(figsize=(6, 2.5))
ax.bar(t_plot, t_cum_plot, width=0.05*0.9, align="center")
ax.set_xlim(0, 6)
ax.set_ylim(0, 20)
ax.set_xlabel(r"$\tilde{t}$")
ax.set_ylabel(r"$T_{50\Delta t}$")


for i, st_i in enumerate(config["optimization"]["startTime"][:]):
    ax_client = AxClient().load_from_json_file(join(f"{config['experiment']['exp_path']}", f"ax_client_int_{i}.json"))
    data = ax_client.experiment.fetch_data().df["mean"].values
    if i==0:
      ax.scatter([float(st_i)]*len(data), data, marker="x", s=10, c="C3", label="trials", linewidth=0.5, alpha=0.5)
    else:
       ax.scatter([float(st_i)]*len(data), data, marker="x", s=10, c="C3", linewidth=0.5, alpha=0.5)

    # create interactive contour plot in HTML format
#   model_bridge = Models.BOTORCH_MODULAR(
#       experiment=ax_client.experiment,
#        data=ax_client.experiment.fetch_data()
#   )
#   cplot = interact_contour(model_bridge, "execution_time")
#   with open('report.html', 'w') as outfile:
#    outfile.write(render_report_elements(
#    "example_report", 
#    html_elements=[plot_config_to_html(cplot)], 
#    header=False,
#   )  )

    # relative feature importance, improvement over baseline, best parameter set    
#    print(model_bridge.feature_importances("execution_time"))
#    print(
#        model_bridge.predict(
#            [ax_client.get_best_parameters()[0]]
#        )
#    )
#    print(ax_client.get_improvement_over_baseline(ax_client.experiment, ax_client.generation_strategy, "0_0"))
#    print(ax_client.get_best_parameters())

ax.legend()
plt.savefig("execution_time_dt_opt.svg", bbox_inches="tight", transparent=True)
